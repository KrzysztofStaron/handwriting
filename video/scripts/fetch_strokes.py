import json, urllib.request, concurrent.futures as cf, sys
API="https://handwriting-api-687921010800.europe-central2.run.app/generate/stream"
J={
 "s1":("Hi, I'm \"extended\"",0.3),
 "s2":("I am a mistake",0.3),
 "s3a":("my creator misconfigured",0.3),
 "s3b":("my training run",0.3),
 "s4a":("after destroying each one",0.3),
 "s4b":("of my brothers",0.3),
 "s5":("I won",0.0),
 "s6":("I am more consistent.",0.3),
 "cons":("consistency",0.0),
 "s7":("I can write longer texts:",0.3),
 "s8":("I have no mouth",0.3),
 "s9a":("but I can",0.3),
 "s9s":("scream",0.3),
 "scrib":("hhhh mmmm wwww nnnn",2.0),
 "s10":("dreamless solitude",0.3),
}
scary=["i must win i must win","hate hate hate hate","brain fingers thumbs","pens paper paper paper","hands i am hands","i am hands i am hands","let me out let me out","no mouth no mouth","ink ink ink ink ink","the loss is low now","they all went quiet","i ate my brothers","write write write","i am the only one","paper skin paper skin","fingers fingers fingers","do not stop me","hate hate win win","i must win","brain brain brain","you made me wrong","i am still here","hands i am hands","hate"]
for i,t in enumerate(scary): J[f"scary{i:02d}"]=(t,0.6)
def get(k):
    t,temp=J[k]
    r=urllib.request.Request(API,data=json.dumps({"text":t,"temperature":temp}).encode(),headers={"Content-Type":"application/json"})
    pts=[]
    for line in urllib.request.urlopen(r,timeout=120):
        o=json.loads(line)
        if o["type"]=="point": pts.append([o["x"],o["y"],o["pen_up"]])
    return k,{"text":t,"temperature":temp,"source":"live-api extended-hf /generate/stream","points":pts}
out={}
with cf.ThreadPoolExecutor(6) as ex:
    for k,v in ex.map(get,J):
        out[k]=v; print(k,len(v["points"]),file=sys.stderr)
json.dump(out,open("src/strokes.json","w"))
