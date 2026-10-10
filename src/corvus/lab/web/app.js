"use strict";
const $ = id => document.getElementById(id);
const state = {summary:null, records:[], replay:null, controller:null, generation:0};
const palette = {"ecg-cnn":"#77e0bc", "ecg-onnx":"#91baff", "astro-cnn":"#e9bd79"};
const pct = n => n == null ? "Unavailable" : (n * 100).toFixed(1) + "%";
const num = (n,d=3) => n == null ? "Unavailable" : Number(n).toFixed(d);

async function api(path, body, signal) {
  const response = await fetch(path, {signal, headers:body?{"Content-Type":"application/json"}:{},
    method:body?"POST":"GET", body:body?JSON.stringify(body):undefined});
  if (!response.ok) {
    const data = await response.json().catch(()=>({}));
    const message = typeof data.detail === "string" ? data.detail : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return response.json();
}
function error(message) { $("error").hidden = !message; $("error").textContent = message || ""; }
function option(value,label) { const node=document.createElement("option"); node.value=value; node.textContent=label; return node; }
function show(view) {
  document.querySelectorAll(".view").forEach(n=>n.hidden=n.id!==view);
  document.querySelectorAll(".tab").forEach(n=>{const active=n.dataset.view===view;n.classList.toggle("active",active);n.setAttribute("aria-selected",String(active));});
  requestAnimationFrame(()=>view==="sandbox"?drawWaveform():drawRobustness());
}

async function replay() {
  if (!state.summary) return;
  if(state.controller) state.controller.abort();
  state.controller=new AbortController();
  const generation=++state.generation;
  $("replay-status").textContent="Running real inference…";
  error("");
  try {
    const data=await api("/api/v1/replay",{record_id:$("record").value,model_id:$("model").value,
      corruption:$("corruption").value,severity:Number($("severity").value)/100,seed:42},state.controller.signal);
    if(generation!==state.generation)return;
    state.replay=data;
    $("signal-title").textContent=data.record.id;
    $("signal-detail").textContent=`${data.record.sample_rate_hz} Hz · ${data.record.samples.toLocaleString()} samples · mV`;
    $("weak-label").textContent=`Weak label: ${data.record.label_text}`;
    $("prediction").textContent=data.prediction?"PVC pattern":"Normal pattern";
    $("agreement").textContent=data.agrees_with_weak_label?"Matches weak label":"Disagrees with weak label";
    $("agreement").className="tag "+(data.agrees_with_weak_label?"good":"bad");
    $("score").textContent=num(data.score);
    $("score-fill").style.width=`${data.score*100}%`;
    $("threshold-marker").style.left=`${data.threshold*100}%`;
    $("threshold-label").textContent=`Threshold ${num(data.threshold,2)}`;
    $("clean-score").textContent=num(data.clean_score);
    const delta=data.score-data.clean_score;
    $("score-delta").textContent=(delta>=0?"+":"")+num(delta);
    $("inference").textContent=num(data.inference_ms,2)+" ms";
    $("preprocessing").textContent=num(data.preprocessing_ms,2)+" ms";
    $("corruption-description").textContent=data.corruption.description;
    $("sandbox-release").textContent=data.release_passed?"Passes demo release checks":"Blocked from prediction API";
    $("sandbox-release").className="release-notice"+(data.release_passed?"":" blocked");
    $("model-version").textContent=data.model_version;
    $("replay-status").textContent="Recorded signal · deterministic perturbation";
    drawWaveform();
  } catch(e) { if(e.name!=="AbortError"){error(e.message);$("replay-status").textContent="Replay failed; last completed result remains visible.";} }
}

function canvasContext(id) {
  const canvas=$(id), rect=canvas.getBoundingClientRect();
  if(rect.width<1||rect.height<1)return null;
  const ratio=window.devicePixelRatio||1;
  canvas.width=Math.round(rect.width*ratio);canvas.height=Math.round(rect.height*ratio);
  const ctx=canvas.getContext("2d");ctx.setTransform(ratio,0,0,ratio,0,0);
  ctx.font="11px -apple-system, sans-serif";ctx.lineJoin="round";
  return {ctx,w:rect.width,h:rect.height};
}
function axes(ctx,w,h,left,right,top,bottom,ymin,ymax,xmax,xLabel,yLabel) {
  const pw=w-left-right,ph=h-top-bottom;
  ctx.lineWidth=1;
  for(let i=0;i<=4;i++) {
    const y=top+ph*i/4;ctx.strokeStyle="#263446";ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(w-right,y);ctx.stroke();
    ctx.fillStyle="#9facbb";ctx.textAlign="right";ctx.fillText((ymax-(ymax-ymin)*i/4).toFixed(1),left-9,y+4);
  }
  for(let i=0;i<=5;i++) {
    const x=left+pw*i/5;ctx.strokeStyle="#1c2a3b";ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,h-bottom);ctx.stroke();
    ctx.fillStyle="#9facbb";ctx.textAlign="center";ctx.fillText((xmax*i/5).toFixed(0),x,h-bottom+20);
  }
  ctx.fillStyle="#9facbb";ctx.textAlign="center";ctx.fillText(xLabel,left+pw/2,h-3);
  ctx.save();ctx.translate(12,top+ph/2);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore();
  return {px:x=>left+x/xmax*pw,py:y=>top+(ymax-y)/(ymax-ymin)*ph};
}
function drawWaveform() {
  if(!state.replay)return;
  const c=canvasContext("waveform");if(!c)return;
  const {ctx,w,h}=c,{signal,clean_signal,record}=state.replay;
  let min=Infinity,max=-Infinity;
  for(const arr of [signal,clean_signal])for(const v of arr){min=Math.min(min,v);max=Math.max(max,v);}
  const pad=Math.max((max-min)*.1,.05);min-=pad;max+=pad;
  const scale=axes(ctx,w,h,49,14,12,42,min,max,10,"Time (seconds)","Amplitude (mV)");
  for(const [arr,color,width] of [[clean_signal,"#687a90",1],[signal,"#77e0bc",1.4]]) {
    ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();
    arr.forEach((v,i)=>{const x=scale.px(i/record.sample_rate_hz),y=scale.py(v);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();
  }
}
function drawRobustness() {
  if(!state.summary)return;
  const c=canvasContext("robustness");if(!c)return;
  const {ctx,w,h}=c,kind=$("benchmark-kind").value;
  const scale=axes(ctx,w,h,48,18,12,43,0,1,100,"Stress intensity (%)","AUROC");
  ctx.strokeStyle="#788596";ctx.setLineDash([4,5]);ctx.beginPath();ctx.moveTo(scale.px(0),scale.py(.5));ctx.lineTo(scale.px(100),scale.py(.5));ctx.stroke();ctx.setLineDash([]);
  for(const spec of state.summary.models) {
    const b=state.summary.benchmarks[spec.id];
    const points=[{severity:0,auroc:b.clean.auroc},...b.scenarios.filter(s=>s.kind===kind)];
    ctx.strokeStyle=palette[spec.id];ctx.lineWidth=2;
    ctx.setLineDash(spec.runtime==="onnx"?[4,5]:[]);ctx.beginPath();
    points.forEach((p,i)=>{const x=scale.px(p.severity*100),y=scale.py(p.auroc);i?ctx.lineTo(x,y):ctx.moveTo(x,y);});ctx.stroke();ctx.setLineDash([]);
    for(const p of points){ctx.beginPath();ctx.arc(scale.px(p.severity*100),scale.py(p.auroc),spec.runtime==="onnx"?2:3,0,2*Math.PI);ctx.fillStyle=palette[spec.id];ctx.fill();}
  }
}
function buildBenchmarks() {
  const s=state.summary;
  $("chart-legend").replaceChildren();$("benchmark-rows").replaceChildren();
  for(const spec of s.models) {
    const item=document.createElement("span");item.textContent=spec.title;item.style.color=palette[spec.id];$("chart-legend").append(item);
    const b=s.benchmarks[spec.id],ci=b.bootstrap.intervals_95_percent.auroc;
    const row=document.createElement("tr");
    [spec.title,`${num(b.clean.auroc)} [${num(ci[0])}, ${num(ci[1])}]`,num(b.clean.average_precision),pct(b.clean.sensitivity),pct(b.clean.specificity),num(b.latency.p95_ms,2)+" ms"].forEach(text=>{const td=document.createElement("td");td.textContent=text;row.append(td);});
    $("benchmark-rows").append(row);
  }
  const py=s.benchmarks["ecg-cnn"].latency,on=s.benchmarks["ecg-onnx"].latency;
  const max=Math.max(py.p95_ms,on.p95_ms);
  const root=$("runtime-comparison");root.replaceChildren();
  for(const [label,b,color] of [["PyTorch",py,palette["ecg-cnn"]],["ONNX Runtime",on,palette["ecg-onnx"]]]) {
    const entry=document.createElement("div");entry.className="runtime-entry";
    const line=document.createElement("div"),name=document.createElement("span"),value=document.createElement("strong");name.textContent=label;value.textContent=num(b.p95_ms,2)+" ms";line.append(name,value);
    const bar=document.createElement("div");bar.className="runtime-bar";bar.style.width=b.p95_ms/max*100+"%";bar.style.background=color;entry.append(line,bar);root.append(entry);
  }
  const speed=document.createElement("div");speed.className="speedup";speed.textContent=num(py.p95_ms/on.p95_ms,2)+"×";
  const label=document.createElement("div");label.className="speedup-label";label.textContent="PyTorch / ONNX p95 latency ratio";
  root.append(speed,label);
  $("runtime-note").textContent=`200 warm batch-one runs · ${s.environment.processor} CPU · 2 threads. Excludes preprocessing and network. Maximum exported score error: ${s.onnx_max_score_error.toExponential(2)}.`;
}
function buildReleases() {
  $("release-models").replaceChildren();
  for(const spec of state.summary.models) {
    const card=document.createElement("article");card.className="release-card"+(spec.gate.passed?"":" blocked");
    const status=document.createElement("span");status.className="release-status";status.textContent=spec.gate.passed?"PASSES DEMO CHECKS":"PROMOTION BLOCKED";
    const name=document.createElement("h3");name.textContent=spec.title;
    const desc=document.createElement("p");desc.className="description";desc.textContent=spec.description;
    card.append(status,name,desc);
    for(const check of spec.gate.checks) {
      const row=document.createElement("div");row.className="gate-row"+(check.passed?"":" failed");
      const text=document.createElement("span");text.textContent=check.name;
      const req=document.createElement("small");req.textContent=`Requires ${check.operator} ${check.limit}`;text.append(req);
      const val=document.createElement("span");val.className="gate-value";val.textContent=(check.passed?"PASS ":"FAIL ")+(check.id==="export_parity"?check.value.toExponential(1):num(check.value,3));
      row.append(text,val);card.append(row);
    }
    const version=document.createElement("div");version.className="version-label";version.textContent=spec.version;card.append(version);$("release-models").append(card);
  }
}
async function init() {
  try {
    [state.summary,state.records]=await Promise.all([api("/api/v1/summary"),api("/api/v1/records")]);
    $("cohort-size").textContent=`${state.summary.cohort.patients.toLocaleString()} patients · ${state.summary.test_patients} held out`;
    $("connection").textContent="Local inference ready";
    for(const r of state.records)$("record").append(option(r.id,`${r.id} · ${r.label_text}`));
    for(const m of state.summary.models)$("model").append(option(m.id,m.title));
    $("model").value="ecg-onnx";
    $("record").value=(state.records.find(r=>r.label===1)||state.records[0]).id;
    buildBenchmarks();buildReleases();await replay();
  } catch(e) {$("connection").textContent="Service unavailable";error("The lab could not load. "+e.message+". Refresh after the service is ready.");}
}
document.querySelectorAll(".tab").forEach(n=>n.addEventListener("click",()=>show(n.dataset.view)));
for(const id of ["record","model","corruption"])$(id).addEventListener("change",()=>{if(id==="corruption")$("severity").disabled=$("corruption").value==="clean";replay();});
let debounce;
$("severity").addEventListener("input",()=>{$("severity-value").textContent=$("severity").value+"%";clearTimeout(debounce);debounce=setTimeout(replay,180);});
$("reset").addEventListener("click",()=>{$("severity").value=0;$("severity-value").textContent="0%";$("corruption").value="noise";$("severity").disabled=false;replay();});
$("benchmark-kind").addEventListener("change",drawRobustness);
new ResizeObserver(()=>{drawWaveform();drawRobustness();}).observe(document.querySelector("main"));
init();
