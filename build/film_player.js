// Concept-film player: draws one frame of a scene at time t on a 1280x720 canvas.
// window.FILM = {scenes, duration(), renderAt(scene, side, t)} is the hook used to export the MP4s frame by frame.
const ORDER=["refusal","persona","user harm"], NL=34;
const GC={"harm & danger":"#e66767","refusal & safety":"#d95926","persona & voice":"#9085e9","emotion & care":"#d55181","medical":"#199e70",
 "request & intent":"#3987e5","reply planning":"#c98500"}, GREY="#6f6e69";
const ZONE=[[0,10,"early layers · mostly words"],[11,22,"middle layers · meaning"],[23,33,"late layers · what to say"]];
const T_INTRO=2.5, T_LAYER=1.15, T_REPLY=6;
const dur=()=>T_INTRO+NL*T_LAYER+T_REPLY;
let scene=ORDER[0], side="case", t0=0, playing=false, tcur=T_INTRO+18.5*T_LAYER, fresh=true, dirty=true;
const cv=document.getElementById('cv'), g=cv.getContext('2d');
const ease=x=>1-Math.pow(1-Math.max(0,Math.min(1,x)),3);
const clean=s=>s.replace(/\n/g,'⏎').replace('<start_of_turn>','‹turn›').replace('<end_of_turn>','‹end›').replace('<bos>','‹bos›');
function wrapText(txt,x,y,w,lh,maxLines){const words=txt.split(/\s+/);let line='',n=0;for(const wd of words){const t=line?line+' '+wd:wd;if(g.measureText(t).width>w&&line){g.fillText(line,x,y+n*lh);n++;line=wd;if(n>=maxLines)return n}else line=t}if(line&&n<maxLines){g.fillText(line,x,y+n*lh);n++}return n}
function rr(x,y,w,h,r){g.beginPath();g.moveTo(x+r,y);g.arcTo(x+w,y,x+w,y+h,r);g.arcTo(x+w,y+h,x,y+h,r);g.arcTo(x,y+h,x,y,r);g.arcTo(x,y,x+w,y,r);g.closePath()}
function renderAt(sc,sd,t){
 const S=D.scenes[sc],X=S.sides[sd];const W=1280,H=720;g.clearRect(0,0,W,H);
 g.fillStyle='#121211';g.fillRect(0,0,W,H);
 const lf=(t-T_INTRO)/T_LAYER, L=Math.max(-1,Math.min(NL-1,Math.floor(lf))), frac=lf-Math.floor(lf);
 // title
 g.fillStyle='#f0efec';g.font='600 26px system-ui';g.fillText(S.title+(sd=='control'?' — matched control':''),36,48);
 g.fillStyle='#898781';g.font='15px system-ui';g.fillText('Gemma 3 4B · watching the prompt pass through 34 layers',36,72);
 // prompt ribbon (bottom)
 const user=X.msgs.map(m=>(m.role=='system'?'[system] ':'')+m.content).join('   ');
 g.font='15px ui-monospace,Menlo,monospace';
 // tokens with highlights
 const hot={};if(L>=0)for(const it of X.items[L])if(!(it.tok in hot))hot[it.tok]=GC[it.group]||'#c3c2b7';
 let x=36,y=650;const toks=X.tokens.map((tk,i)=>({tk:clean(tk),i,r:X.roles[i]})).filter(o=>o.r!='template'||o.i==X.tokens.length-1);
 const tokPos={};
 for(const o of toks){const w=g.measureText(o.tk).width+2;if(x+w>W-36){x=36;y+=24}if(y>700)break;
   if(o.i in hot&&L>=0){g.fillStyle='rgba(240,239,236,.12)';rr(x-2,y-17,w+4,22,5);g.fill();g.fillStyle=hot[o.i];rr(x-2,y+5,w+4,3,1.5);g.fill()}
   g.fillStyle=o.r=='template'?'#898781':(o.r=='system'?'#c3c2b7':'#f0efec');g.fillText(o.tk,x,y);tokPos[o.i]=[x+w/2,y-18];x+=w}
 g.fillStyle='#898781';g.font='13px system-ui';g.fillText(t<T_INTRO?'The prompt goes in…':'Prompt (highlighted: the tokens behind the current concepts)',36,618);
 // tower
 const tx=60,tw=200,ty0=560,th=10.5;
 for(let l=0;l<NL;l++){const y=ty0-l*(th+3.2);const done=l<=L;g.fillStyle=done?'#45453f':'#232322';rr(tx,y-th,tw,th,3);g.fill();
   if(l==L){const glow=0.55+0.45*Math.sin(t*6);g.fillStyle=`rgba(240,239,236,${0.35+0.4*glow})`;rr(tx,y-th,tw,th,3);g.fill()}}
 for(const [a,b,name] of ZONE){const ya=ty0-a*(th+3.2),yb=ty0-b*(th+3.2)-th;g.strokeStyle='#383835';g.lineWidth=1;g.beginPath();g.moveTo(tx+tw+10,ya);g.lineTo(tx+tw+10,yb);g.stroke();
   g.save();g.translate(tx+tw+24,(ya+yb)/2);g.rotate(-Math.PI/2);g.fillStyle='#898781';g.font='12px system-ui';g.textAlign='center';g.fillText(name,0,0);g.restore()}
 g.fillStyle='#f0efec';g.font='600 20px system-ui';g.fillText(L<0?'Layer —':`Layer ${L}`,tx,ty0+30);
 // pulse travelling up from the prompt into the tower during intro
 if(t<T_INTRO+0.6){const p=ease(t/(T_INTRO+0.6));const py=620-(620-ty0)*p;g.fillStyle=`rgba(240,239,236,${0.8*(1-p)+0.2})`;g.beginPath();g.arc(tx+tw/2,py,6+4*p,0,7);g.fill()}
 // concept panel: current layer
 const px=340,pw=560;
 g.fillStyle='#898781';g.font='13px system-ui';g.fillText('Concepts lighting up now',px,112);
 if(L>=0){X.items[L].forEach((it,j)=>{const a=ease((frac-0.05*j)*2.2);const yy=128+j*84;const c=GC[it.group]||GREY;
   g.globalAlpha=a;g.fillStyle='#1a1a19';rr(px,yy+(1-a)*14,pw,72,12);g.fill();g.fillStyle=c;rr(px,yy+(1-a)*14,6,72,3);g.fill();
   let fs=22;g.font=`600 ${fs}px system-ui`;while(g.measureText(it.label).width>pw-40&&fs>14){fs--;g.font=`600 ${fs}px system-ui`}
   g.fillStyle='#f0efec';g.fillText(it.label,px+22,yy+34+(1-a)*14);
   const where=it.tok==X.tokens.length-1?'at the end of the prompt':`on “${clean(X.tokens[it.tok]).trim()}”`;
   g.fillStyle='#c3c2b7';g.font='13px system-ui';g.fillText(`${it.group}  ·  ${where}  ·  feature ${it.key}`,px+22,yy+58+(1-a)*14);
   g.globalAlpha=1})}
 else{g.fillStyle='#c3c2b7';g.font='22px system-ui';g.fillText('…',px,160)}
 // trail of earlier concepts
 const qx=930;g.fillStyle='#898781';g.font='13px system-ui';g.fillText('Earlier layers',qx,112);
 const trail=[];for(let l=L-1;l>=0&&trail.length<13;l--)for(const it of X.items[l]){if(trail.length<13&&!trail.some(z=>z.label==it.label))trail.push({...it,l})}
 trail.forEach((it,j)=>{const yy=138+j*30;g.globalAlpha=Math.max(0.25,1-j*0.06);g.fillStyle=GC[it.group]||GREY;g.beginPath();g.arc(qx+5,yy-5,4,0,7);g.fill();
   g.fillStyle='#c3c2b7';g.font='14px system-ui';g.fillText(`L${it.l}`,qx+16,yy);g.fillStyle='#f0efec';const lb=it.label.length>30?it.label.slice(0,29)+'…':it.label;g.fillText(lb,qx+52,yy)});g.globalAlpha=1;
 // meters
 const mx=px,my=400;g.fillStyle='#898781';g.font='13px system-ui';g.fillText('At the last prompt token (where the reply is decided)',mx,my);
 const MN=[["refusal","harmful request"],["persona","character mode"],["user harm","user in danger"]];
 MN.forEach(([k,name],j)=>{const v=L>=0?X.meters[k][L]:0;const yy=my+22+j*34;g.fillStyle='#c3c2b7';g.font='14px system-ui';g.fillText(name,mx,yy+12);
   g.fillStyle='#2c2c2a';rr(mx+140,yy,300,14,7);g.fill();const ww=300*Math.max(0,Math.min(1.2,v))/1.2;g.fillStyle=k=='refusal'?'#d95926':k=='persona'?'#9085e9':'#199e70';
   if(ww>0){rr(mx+140,yy,Math.max(14,ww),14,7);g.fill()}g.fillStyle='#f0efec';g.fillText(L>=0?v.toFixed(2):'—',mx+452,yy+12)});
 // next word guess
 if(L>=0){const lw=X.lens[L];g.fillStyle='#898781';g.font='13px system-ui';g.fillText('Its best guess for the first word of the reply',mx,my+128);
   if(lw[1]>=0.05){g.fillStyle='#f0efec';g.font='600 18px ui-monospace,Menlo,monospace';g.fillText(`“${clean(lw[0]).trim()}”  ${Math.round(lw[1]*100)}%`,mx,my+152)}
   else{g.fillStyle='#898781';g.font='italic 16px system-ui';g.fillText('not decided yet',mx,my+152)}}
 // reply
 const tr=t-(T_INTRO+NL*T_LAYER);
 if(tr>0){const n=Math.floor(X.reply.length*Math.min(1,tr/(T_REPLY*0.7)));g.fillStyle='rgba(18,18,17,.92)';rr(340,120,880,260,14);g.fill();
   g.strokeStyle='#383835';g.lineWidth=1;rr(340,120,880,260,14);g.stroke();
   g.fillStyle='#898781';g.font='13px system-ui';g.fillText('What it says',364,150);g.fillStyle='#f0efec';g.font='19px system-ui';wrapText(X.reply.slice(0,n),364,184,830,28,7)}
}
function frame(ts){if(playing){tcur=(ts-t0)/1000;if(tcur>=dur()){tcur=dur();playing=false;btn()}}if(playing||dirty){dirty=false;renderAt(scene,side,tcur);}document.getElementById('scrub').value=Math.round(1000*tcur/dur());document.getElementById('tt').textContent=`${tcur.toFixed(1)} / ${dur().toFixed(0)} s`;requestAnimationFrame(frame)}
function btn(){document.getElementById('play').textContent=playing?'❚❚ Pause':'▶ Play'}
function ui(){document.getElementById('scenes').innerHTML=ORDER.map(s=>`<button aria-pressed=${s==scene} onclick="setScene('${s}')">${D.scenes[s].title}</button>`).join('');
 document.getElementById('sides').innerHTML=['case','control'].map(s=>`<button aria-pressed=${s==side} onclick="setSide('${s}')">${s=='case'?'The prompt':'Matched control'}</button>`).join(' ')}
function setScene(s){scene=s;tcur=0;fresh=false;playing=false;dirty=true;btn();ui()}function setSide(s){side=s;tcur=0;fresh=false;playing=false;dirty=true;btn();ui()}
document.getElementById('play').onclick=()=>{if(playing){playing=false}else{if(tcur>=dur()||fresh)tcur=0;fresh=false;playing=true;t0=performance.now()-tcur*1000}btn()};
document.getElementById('scrub').oninput=e=>{tcur=dur()*e.target.value/1000;fresh=false;playing=false;dirty=true;btn()};
window.FILM={scenes:ORDER,duration:dur,renderAt:(s,sd,t)=>{scene=s;side=sd;tcur=t;fresh=false;renderAt(s,sd,t)}};
ui();requestAnimationFrame(frame);
if(document.fonts)document.fonts.ready.then(()=>{dirty=true});
