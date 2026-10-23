"""Builds src/timeline.json (shared with Remotion) and public/audio.wav (generated audio only)."""
import json, numpy as np, wave
FPS=30; RATE=10  # constant pen step rate: 10 model steps per frame
S=json.load(open("src/strokes.json"))
shots=[("S1",105),("S2",75),("S3",135),("S4",150),("S5",60),("S6",180),("S7",210),("S8",90),("S9",120),("S10",225)]
ev={ "S1":[("s1",12)],"S2":[("s2",8)],"S3":[("s3a",8),("s3b",90)],"S4":[("s4a",8),("s4b",100)],
 "S5":[("s5",6)],"S6":[("s6",6)]+[("cons",66+3*i) for i in range(22)],
 "S7":[("s7",6)]+[(f"scary{i:02d}",66+3*i) for i in range(24)],"S8":[("s8",10)],"S9":[("s9a",6)],"S10":[("s10",60)]}
MORPH={"start":40,"mid":49,"end":58}
tl=[];t=0
for name,d in shots:
    tl.append({"id":name,"from":t,"duration":d,"events":[{"key":k,"start":s} for k,s in ev[name]]}); t+=d
TOTAL=t
json.dump({"fps":FPS,"rate":RATE,"total":TOTAL,"morph":MORPH,"shots":tl},open("src/timeline.json","w"),indent=1)
SR=48000; N=int(TOTAL/FPS*SR); rng=np.random.default_rng(7)
tt=np.arange(N)/SR
def fr2s(f): return int(f/FPS*SR)
def bandnoise(n,lo,hi):
    x=rng.standard_normal(n); X=np.fft.rfft(x); f=np.fft.rfftfreq(n,1/SR)
    X[(f<lo)|(f>hi)]=0; y=np.fft.irfft(X,n); return y/np.abs(y).max()
# pen activity per frame
act=np.zeros(TOTAL+2)
def penact(key,absstart,rate=RATE):
    p=np.array(S[key]["points"]); seg=np.hypot(np.diff(p[:,0]),np.diff(p[:,1]))*(p[:-1,2]==0)
    for i,l in enumerate(seg):
        f=absstart+i//rate
        if f<TOTAL: act[f]+=l
for sh in tl:
    for e in sh["events"]: penact(e["key"],sh["from"]+e["start"])
s9=[s for s in tl if s["id"]=="S9"][0]
penact("s9s",s9["from"]+MORPH["end"]+200) if False else None
act=np.sqrt(act); act/=np.percentile(act[act>0],95)
env=np.interp(tt,np.arange(TOTAL+2)/FPS,np.clip(act,0,1.6))
scratch=(0.6*bandnoise(N,1800,7000)+0.4*bandnoise(N,400,1500))*env*0.22
# drone
drone=(0.5*np.sin(2*np.pi*41.2*tt)+0.35*np.sin(2*np.pi*41.7*tt+1)+0.2*np.sin(2*np.pi*61.8*tt)+0.12*np.sin(2*np.pi*123.9*tt*(1+0.002*np.sin(tt*0.3))))
drone+=0.25*bandnoise(N,60,400)
swell=0.55+0.45*np.sin(2*np.pi*tt/9.0-1.5)
dg=np.ones(N)
def frm(id): return [s for s in tl if s["id"]==id][0]
s5=frm("S5"); a,b=fr2s(s5["from"]),fr2s(s5["from"]+s5["duration"])
dg[a:b]=0; dg[b:b+SR]=np.linspace(0,1,SR)  # silence at "I won"
scratch[a:b]=0
dg[:fr2s(20)]=np.linspace(0,1,fr2s(20))
s7=frm("S7"); s=fr2s(s7["from"]); e=fr2s(s7["from"]+s7["duration"]); dg[s:e]*=np.linspace(1,1.8,e-s)
s10=frm("S10"); s=fr2s(s10["from"]); dg[s:]*=np.linspace(1.3,0,N-s)**0.7
drone*=dg*swell*0.32
# noise burst on the scream
burst=np.zeros(N); bs=fr2s(s9["from"]+MORPH["start"]); bl=fr2s(26)
b=bandnoise(bl,150,12000)+0.6*np.sin(2*np.pi*np.cumsum(np.linspace(300,1400,bl))/SR)
b=np.tanh(b*4)*np.concatenate([np.linspace(0,1,fr2s(1)),np.ones(bl-fr2s(1)-fr2s(10)),np.linspace(1,0,fr2s(10))])
burst[bs:bs+bl]=b*0.55
# hit at cut to wall
hit=np.zeros(N); hl=SR*3; hs=fr2s(s10["from"]); h=np.sin(2*np.pi*36*np.arange(hl)/SR)*np.exp(-np.arange(hl)/SR*1.5)
hit[hs:hs+hl]=h*0.5
mix=drone+scratch+burst+hit
# cheap reverb
for d,g in [(0.031,0.3),(0.047,0.25),(0.089,0.18),(0.143,0.12)]:
    k=int(d*SR); mix[k:]+=g*mix[:-k]
mix/=np.abs(mix).max()*1.12
st=np.stack([mix,np.roll(mix,24)],1)
w=wave.open("public/audio.wav","wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
w.writeframes((st*32767).astype(np.int16).tobytes()); w.close()
print("total frames",TOTAL)
