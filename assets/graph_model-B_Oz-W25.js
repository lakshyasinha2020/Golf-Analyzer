import{fs as g,dq as h,a3 as b,a as Dn,b as An,cz as y,cx as Pe,h as Cn,i as zn,j as Pn,k as Vn,l as Ln,m as Fn,n as Rn,o as Bn,f8 as Q,p as jn,q as qn,dH as be,fU as w,d0 as oe,r as Kn,cX as K,s as Wn,dr as pt,d2 as Ce,I as Hn,B as Gn,bS as Un,cc as Xn,ct as W,fi as I,g5 as We,v as Yn,aq as Qn,w as Zn,x as Jn,y as Mn,D as er,dY as Nt,E as tr,gq as Ve,J as sr,L as nr,M as rr,O as ar,P as or,Q as ir,S as ur,T as cr,V as pr,W as lr,X as mr,_ as hr,$ as dr,cA as G,af as fr,cQ as tt,bK as yr,dG as X,g$ as Xt,ab as gr,cy as br,ag as Nr,aT as Tr,a$ as wr,ft as Yt,dT as Tt,cr as se,gy as R,gi as At,fv as Qt,fX as L,gk as wt,ah as $r,ai as Sr,aj as _r,cg as vr,cU as Fe,an as kr,au as Er,av as xr,aw as Or,az as Ir,aA as Dr,aB as Ar,aC as Cr,aH as zr,aI as Pr,aJ as Vr,eQ as Fs,aE as Lr,aK as Fr,aL as Rr,b6 as Br,bU as jr,dw as he,gu as P,aN as qr,aO as Kr,aP as Wr,g8 as Se,bJ as Hr,aU as Gr,aV as Ur,aY as Xr,a_ as Yr,cC as Ne,gZ as $t,d4 as Le,f6 as Qr,cd as _e,b0 as Zr,b1 as Jr,b2 as Mr,b4 as ea,ba as ta,bc as sa,bd as na,bg as ra,bV as aa,di as oa,bk as ia,bm as ua,bn as ca,bo as pa,bp as la,br as ma,bt as ha,bB as da,bD as fa,bE as ya,bL as ga,ce as ds,bP as ba,bQ as Na,bR as Ta,bT as wa,ak as $a,ax as Sa,b$ as _a,c2 as va,gm as ka,bf as Ea,c5 as xa,cb as Oa,cD as Re,eM as Be,f7 as je,gS as Rs,cf as Ia,ch as Da,ck as Aa,cm as Ca,cl as za,gY as Pa,cH as ie,gC as ae,cj as xt,cI as Va,bI as La,b_ as Fa,at as Ra,ey as Ba,gA as ve,K as ja,g3 as Zt,cw as Jt,d9 as Bs,ar as fs,ef as Mt,gL as js,ee as es,Y as qa,Z as Ka,as as ys,cq as gs,R as Wa,am as Ha,bC as Ga,b7 as Ua,fn as Xa,b9 as Ya,fp as Qa,b8 as Za,fo as Ja,bx as Ma,bz as eo,ci as to,dE as qs,fP as ts,bW as so,bX as no,bY as ro,bZ as ao,c6 as oo,c7 as io,c8 as uo,c3 as co,H as po,cS as lo,cT as mo,dm as ho,ds as fo,dz as yo,dA as Ks,dM as go,em as bo,en as No,eo as Ws,ep as To,eu as wo,eA as Hs,eG as $o,eO as So,f2 as _o,f3 as vo,fh as ko,fN as Eo,fO as xo,fS as Oo,gX as Io,a2 as Gs,dO as Ue,ek as Us,ap as bs,gF as Do,bb as Ao,dI as Co,e1 as zo,e_ as Po,fc as Vo,fw as Lo,fz as Fo,fQ as Ro,go as Bo,eZ as me,eT as lt}from"./shared-BOi8COmy.js";import{g as jo}from"./index-D5q4OXnS.js";/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qo(s){const t={x:h(s,"x","acos")};return b.runKernel(Dn,t)}const Ko=g({acos_:qo});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wo(s){const t={x:h(s,"x","acosh")};return b.runKernel(An,t)}const Ho=g({acosh_:Wo});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Go(s){y(Array.isArray(s),()=>"The argument passed to tf.addN() must be a list of tensors"),y(s.length>=1,()=>`Must pass at least one tensor to tf.addN(), but got ${s.length}`);const e=s.map((r,a)=>h(r,`tensors${a}`,"addN")),t=e[0];e.forEach(r=>{if(r.dtype!==t.dtype)throw new Error("All tensors passed to tf.addN() must have the same dtype")}),e.forEach(r=>{if(!Pe(r.shape,t.shape))throw new Error("All tensors passed to tf.addN() must have the same shape")});const n=e;return b.runKernel(Cn,n)}const Uo=g({addN_:Go});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xo(s,e=null,t=!1){const r={x:h(s,"x","all","bool")},a={axis:e,keepDims:t};return b.runKernel(zn,r,a)}const Yo=g({all_:Xo});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qo(s,e=null,t=!1){const r={x:h(s,"x","any","bool")},a={axis:e,keepDims:t};return b.runKernel(Pn,r,a)}const Zo=g({any_:Qo});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Jo(s,e=0){const n={x:h(s,"x","argMax")},r={axis:e};return b.runKernel(Vn,n,r)}const Mo=g({argMax_:Jo});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ei(s,e=0){const n={x:h(s,"x","argMin")},r={axis:e};return b.runKernel(Ln,n,r)}const ti=g({argMin_:ei});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function si(s){const t={x:h(s,"x","asin")};return b.runKernel(Fn,t)}const ni=g({asin_:si});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ri(s){const t={x:h(s,"x","asinh")};return b.runKernel(Rn,t)}const ai=g({asinh_:ri});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oi(s){const t={x:h(s,"x","atan")};return b.runKernel(Bn,t)}const ii=g({atan_:oi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ui(s,e){let t=h(s,"a","atan2"),n=h(e,"b","atan2");[t,n]=Q(t,n);const r={a:t,b:n};return b.runKernel(jn,r)}const ci=g({atan2_:ui});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pi(s){const t={x:h(s,"x","atanh")};return b.runKernel(qn,t)}const li=g({atanh_:pi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mi(s,e,t,n,r){const a=h(s,"x","avgPool","float32"),i=1;y(be(t,i),()=>`Error in avgPool: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`);let o=a,u=!1;a.rank===3&&(u=!0,o=w(a,[1,a.shape[0],a.shape[1],a.shape[2]])),y(o.rank===4,()=>`Error in avgPool: x must be rank 4 but got rank ${o.rank}.`),oe("avgPool",n,r);const p={x:o},l={filterSize:e,strides:t,pad:n,dimRoundingMode:r};let m=b.runKernel(Kn,p,l);return m=K(m,a.dtype),u?w(m,[m.shape[1],m.shape[2],m.shape[3]]):m}const Xs=g({avgPool_:mi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hi(s,e,t,n,r,a="NDHWC"){const i=h(s,"x","avgPool3d","float32");let o=i,u=!1;i.rank===4&&(u=!0,o=w(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),y(o.rank===5,()=>`Error in avgPool3d: x must be rank 5 but got rank ${o.rank}.`),y(a==="NDHWC",()=>`Error in avgPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),y(typeof t=="number"&&t>0||Array.isArray(t)&&t[0]>0&&t[1]>0&&t[2]>0,()=>`Error in avgPool3d: Stride must be > 0, but got '${t}'`),oe("avgPool3d",n,r);const p={x:o},l={filterSize:e,strides:t,pad:n,dimRoundingMode:r,dataFormat:a};let m=b.runKernel(Wn,p,l);return m=K(m,o.dtype),u?w(m,[m.shape[1],m.shape[2],m.shape[3],m.shape[4]]):m}const di=g({avgPool3d_:hi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fi(s,e=0){y(s.length>=1,()=>"Pass at least one tensor to concat");const t=pt(s,"tensors","concat","string_or_numeric");if(t[0].dtype==="complex64"&&t.forEach(a=>{if(a.dtype!=="complex64")throw new Error(`Cannot concatenate complex64 tensors with a tensor
          with dtype ${a.dtype}. `)}),t.length===1)return Ce(t[0]);const n=t,r={axis:e};return b.runKernel(Hn,n,r)}const Y=g({concat_:fi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yi(s,e,t=!1,n=!1){let r=h(s,"a","matMul"),a=h(e,"b","matMul");[r,a]=Q(r,a);const i={a:r,b:a},o={transposeA:t,transposeB:n};return b.runKernel(Gn,i,o)}const V=g({matMul_:yi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gi(s,e,t){const n=h(s,"x","slice","string_or_numeric");if(n.rank===0)throw new Error("Slicing scalar is not possible");const r={x:n},a={begin:e,size:t};return b.runKernel(Un,r,a)}const F=g({slice_:gi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bi(s){const t={x:h(s,"x","tanh","float32")};return b.runKernel(Xn,t)}const Ct=g({tanh_:bi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ni(s,e,t,n,r,a){const i=h(s,"forgetBias","basicLSTMCell"),o=h(e,"lstmKernel","basicLSTMCell"),u=h(t,"lstmBias","basicLSTMCell"),p=h(n,"data","basicLSTMCell"),l=h(r,"c","basicLSTMCell"),m=h(a,"h","basicLSTMCell"),d=Y([p,m],1),f=V(d,o),N=W(f,u),$=N.shape[0],T=N.shape[1]/4,S=[$,T],O=F(N,[0,0],S),C=F(N,[0,T],S),_=F(N,[0,T*2],S),k=F(N,[0,T*3],S),v=W(I(We(O),Ct(C)),I(l,We(W(i,_)))),E=I(Ct(v),We(k));return[v,E]}const Ti=g({basicLSTMCell_:Ni});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wi(s,e,t){const n=h(s,"x","batchToSpaceND"),r=e.reduce((o,u)=>o*u);y(n.rank>=1+e.length,()=>`input rank is ${n.rank} but should be > than blockShape.length ${e.length}`),y(t.length===e.length,()=>`crops.length is ${t.length} but should be equal to blockShape.length  ${e.length}`),y(n.shape[0]%r===0,()=>`input tensor batch is ${n.shape[0]} but is not divisible by the product of the elements of blockShape ${e.join(" * ")} === ${r}`);const a={x:n},i={blockShape:e,crops:t};return b.runKernel(Yn,a,i)}const Ys=g({batchToSpaceND_:wi});function $i(s){let e;return s.rank===0||s.rank===1?e=w(s,[1,1,1,s.size]):s.rank===2?e=w(s,[1,1,s.shape[0],s.shape[1]]):s.rank===3?e=w(s,[1,s.shape[0],s.shape[1],s.shape[2]]):e=s,e}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Si(s,e,t,n,r,a){a==null&&(a=.001);const i=h(s,"x","batchNorm"),o=h(e,"mean","batchNorm"),u=h(t,"variance","batchNorm");let p;r!=null&&(p=h(r,"scale","batchNorm"));let l;n!=null&&(l=h(n,"offset","batchNorm")),y(o.rank===u.rank,()=>"Batch normalization gradient requires mean and variance to have equal ranks."),y(l==null||o.rank===l.rank,()=>"Batch normalization gradient requires mean and offset to have equal ranks."),y(p==null||o.rank===p.rank,()=>"Batch normalization gradient requires mean and scale to have equal ranks.");const d={x:$i(i),scale:p,offset:l,mean:o,variance:u},f={varianceEpsilon:a},N=b.runKernel(Qn,d,f);return w(N,i.shape)}const St=g({batchNorm_:Si});function _i(s,e,t,n,r,a){const i=h(s,"x","batchNorm"),o=h(e,"mean","batchNorm"),u=h(t,"variance","batchNorm");let p;r!=null&&(p=h(r,"scale","batchNorm"));let l;return n!=null&&(l=h(n,"offset","batchNorm")),y(i.rank===2,()=>`Error in batchNorm2D: x must be rank 2 but got rank ${i.rank}.`),y(o.rank===2||o.rank===1,()=>`Error in batchNorm2D: mean must be rank 2 or rank 1 but got rank ${o.rank}.`),y(u.rank===2||u.rank===1,()=>`Error in batchNorm2D: variance must be rank 2 or rank 1 but got rank ${u.rank}.`),p!=null&&y(p.rank===2||p.rank===1,()=>`Error in batchNorm2D: scale must be rank 2 or rank 1 but got rank ${p.rank}.`),l!=null&&y(l.rank===2||l.rank===1,()=>`Error in batchNorm2D: offset must be rank 2 or rank 1 but got rank ${l.rank}.`),St(i,o,u,l,p,a)}const vi=g({batchNorm2d_:_i});function ki(s,e,t,n,r,a){const i=h(s,"x","batchNorm"),o=h(e,"mean","batchNorm"),u=h(t,"variance","batchNorm");let p;r!=null&&(p=h(r,"scale","batchNorm"));let l;return n!=null&&(l=h(n,"offset","batchNorm")),y(i.rank===3,()=>`Error in batchNorm3D: x must be rank 3 but got rank ${i.rank}.`),y(o.rank===3||o.rank===1,()=>`Error in batchNorm3D: mean must be rank 3 or rank 1 but got rank ${o.rank}.`),y(u.rank===3||u.rank===1,()=>`Error in batchNorm3D: variance must be rank 3 or rank 1 but got rank ${u.rank}.`),p!=null&&y(p.rank===3||p.rank===1,()=>`Error in batchNorm3D: scale must be rank 3 or rank 1 but got rank ${p.rank}.`),l!=null&&y(l.rank===3||l.rank===1,()=>`Error in batchNorm3D: offset must be rank 3 or rank 1 but got rank ${l.rank}.`),St(i,o,u,l,p,a)}const Ei=g({batchNorm3d_:ki});function xi(s,e,t,n,r,a){const i=h(s,"x","batchNorm"),o=h(e,"mean","batchNorm"),u=h(t,"variance","batchNorm");let p;r!=null&&(p=h(r,"scale","batchNorm"));let l;return n!=null&&(l=h(n,"offset","batchNorm")),y(i.rank===4,()=>`Error in batchNorm4D: x must be rank 4 but got rank ${i.rank}.`),y(o.rank===4||o.rank===1,()=>`Error in batchNorm4D: mean must be rank 4 or rank 1 but got rank ${o.rank}.`),y(u.rank===4||u.rank===1,()=>`Error in batchNorm4D: variance must be rank 4 or rank 1 but got rank ${u.rank}.`),p!=null&&y(p.rank===4||p.rank===1,()=>`Error in batchNorm4D: scale must be rank 4 or rank 1 but got rank ${p.rank}.`),l!=null&&y(l.rank===4||l.rank===1,()=>`Error in batchNorm4D: offset must be rank 4 or rank 1 but got rank ${l.rank}.`),St(i,o,u,l,p,a)}const Oi=g({batchNorm4d_:xi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ii(s,e,t){const n=h(s,"x","bincount"),r=h(e,"weights","bincount");y(n.dtype==="int32",()=>`Error in bincount: input dtype must be int32, but got ${n.dtype}`),y(t>=0,()=>`size must be non-negative, but got ${t}.`),y(r.size===n.size||r.size===0,()=>`Error in bincount: weights must have the same size as input or0-length, but got input shape: ${n.shape}, weights shape: ${r.shape}.`);const a={x:n,weights:r},i={size:t};return b.runKernel(Zn,a,i)}const Qs=g({bincount_:Ii});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Di(s,e){const t=h(s,"x","bitwiseAnd"),n=h(e,"y","bitwiseAnd");if(!Pe(t.shape,n.shape))throw new Error(`BitwiseAnd: Tensors must have the same shape. x: ${t.shape}, y: ${n.shape}`);if(t.dtype!=="int32"||n.dtype!=="int32")throw new Error(`BitwiseAnd: Only supports 'int32' values in tensor, found type of x: ${t.dtype} and type of y: ${n.dtype}`);const r={a:t,b:n};return b.runKernel(Jn,r)}const Ai=g({bitwiseAnd_:Di});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ci(s,e){const t=h(s,"s0","broadcastArgs","int32"),n=h(e,"s1","broadcastArgs","int32");if(t.rank!==1)throw new Error(`broadcastArgs(): first input must be a vector (rank=1). Has rank ${t.rank}`);if(n.rank!==1)throw new Error(`broadcastArgs(): second input must be a vector (rank=1). Has rank ${n.rank}`);const r={s0:t,s1:n};return b.runKernel(Mn,r)}const zi=g({broadcastArgs_:Ci});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pi(s){const t={x:h(s,"x","ceil","float32")};return b.runKernel(er,t)}const Vi=g({ceil_:Pi});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Li(s,e,t){const n=h(s,"x","clipByValue");if(y(e<=t,()=>`Error in clip: min (${e}) must be less than or equal to max (${t}).`),e===t)return Nt(n.shape,e,n.dtype);const r={x:n},a={clipValueMin:e,clipValueMax:t};return b.runKernel(tr,r,a)}const Fi=g({clipByValue_:Li});function Ri(s){return Y(s,0)}const Bi=g({concat1d_:Ri});function ji(s,e){return Y(s,e)}const qi=g({concat2d_:ji});function Ki(s,e){return Y(s,e)}const Wi=g({concat3d_:Ki});function Hi(s,e){return Y(s,e)}const Gi=g({concat4d_:Hi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ui(s,e,t,n,r="NHWC",a=[1,1],i){const o=h(s,"x","conv2d","float32"),u=h(e,"filter","conv2d","float32");let p=o,l=!1;o.rank===3&&(l=!0,p=w(o,[1,o.shape[0],o.shape[1],o.shape[2]])),y(p.rank===4,()=>`Error in conv2d: input must be rank 4, but got rank ${p.rank}.`),y(u.rank===4,()=>`Error in conv2d: filter must be rank 4, but got rank ${u.rank}.`),oe("conv2d",n,i);const m=r==="NHWC"?p.shape[3]:p.shape[1];y(m===u.shape[2],()=>`Error in conv2d: depth of input (${m}) must match input depth for filter ${u.shape[2]}.`),y(be(t,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${t} and dilations '${a}'`),y(Ve(a),()=>"Error in conv2D: Dilated rates should be larger than 0."),y(Ve(t),()=>"Error in conv2D: Strides should be larger than 0.");const d={x:p,filter:u},f={strides:t,pad:n,dataFormat:r,dilations:a,dimRoundingMode:i},N=b.runKernel(sr,d,f);return l?w(N,[N.shape[1],N.shape[2],N.shape[3]]):N}const _t=g({conv2d_:Ui});function Xi(s,e,t,n,r="NWC",a=1,i){const o=h(s,"x","conv1d"),u=h(e,"filter","conv1d");let p=o,l=!1;o.rank===2&&(l=!0,p=w(o,[1,o.shape[0],o.shape[1]])),y(p.rank===3,()=>`Error in conv1d: input must be rank 3, but got rank ${p.rank}.`),y(u.rank===3,()=>`Error in conv1d: filter must be rank 3, but got rank ${u.rank}.`),oe("conv1d",n,i),y(p.shape[2]===u.shape[1],()=>`Error in conv1d: depth of input (${p.shape[2]}) must match input depth for filter ${u.shape[1]}.`),y(be(t,a),()=>`Error in conv1D: Either stride or dilation must be 1. Got stride ${t} and dilation '${a}'`),y(Ve(a),()=>"Error in conv1D: Dilated rates should be larger than 0."),y(Ve(t),()=>"Error in conv1D: Stride should be larger than 0."),y(r==="NWC",()=>`Error in conv1d: got dataFormat of ${r} but only NWC is currently supported.`);const m=w(u,[1,u.shape[0],u.shape[1],u.shape[2]]),d=w(p,[p.shape[0],1,p.shape[1],p.shape[2]]),T=_t(d,m,[1,t],n,"NHWC",[1,a],i);return l?w(T,[T.shape[2],T.shape[3]]):w(T,[T.shape[0],T.shape[2],T.shape[3]])}const Yi=g({conv1d_:Xi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Qi(s,e,t,n,r,a="NHWC",i){y(s.length===e.rank,()=>`Length of inShape (${s.length}) and rank of dy (${e.rank}) must match`);let o=s,u=e,p=!1;e.rank===3&&(p=!0,u=w(e,[1,e.shape[0],e.shape[1],e.shape[2]]),o=[1,s[0],s[1],s[2]]),y(o.length===4,()=>`Error in conv2dDerInput: inShape must be length 4, but got length ${o.length}.`),y(u.rank===4,()=>`Error in conv2dDerInput: dy must be rank 4, but got rank ${u.rank}`),y(t.rank===4,()=>`Error in conv2dDerInput: filter must be rank 4, but got rank ${t.rank}`);const l=a==="NHWC"?o[3]:o[1],m=a==="NHWC"?u.shape[3]:u.shape[1];y(l===t.shape[2],()=>`Error in conv2dDerInput: depth of input (${l}) must match input depth for filter ${t.shape[2]}.`),y(m===t.shape[3],()=>`Error in conv2dDerInput: depth of output (${m}) must match output depth for filter ${t.shape[3]}.`),oe("conv2dDerInput",r,i);const d={dy:u,filter:t},f={strides:n,pad:r,dataFormat:a,dimRoundingMode:i,inputShape:o},N=b.runKernel(nr,d,f);return p?w(N,[N.shape[1],N.shape[2],N.shape[3]]):N}const Zs=g({conv2DBackpropInput_:Qi});function Zi(s,e,t,n,r,a){const i=h(s,"x","conv2dTranspose"),o=h(e,"filter","conv2dTranspose");return Zs(t,i,o,n,r,"NHWC",a)}const Ji=g({conv2dTranspose_:Zi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mi(s,e,t,n,r="NDHWC",a=[1,1,1]){const i=h(s,"x","conv3d"),o=h(e,"filter","conv3d");let u=i,p=!1;i.rank===4&&(p=!0,u=w(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),y(u.rank===5,()=>`Error in conv3d: input must be rank 5, but got rank ${u.rank}.`),y(o.rank===5,()=>`Error in conv3d: filter must be rank 5, but got rank ${o.rank}.`),y(u.shape[4]===o.shape[3],()=>`Error in conv3d: depth of input (${u.shape[4]}) must match input depth for filter ${o.shape[3]}.`),y(be(t,a),()=>`Error in conv3D: Either strides or dilations must be 1. Got strides ${t} and dilations '${a}'`),y(r==="NDHWC",()=>`Error in conv3d: got dataFormat of ${r} but only NDHWC is currently supported.`),y(Ve(a),()=>"Error in conv3D: Dilated rates should be larger than 0."),y(Ve(t),()=>"Error in conv3D: Strides should be larger than 0.");const l={x:u,filter:o},m={strides:t,pad:n,dataFormat:r,dilations:a},d=b.runKernel(rr,l,m);return p?w(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const eu=g({conv3d_:Mi});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tu(s,e,t,n,r){y(s.length===e.rank,()=>`Length of inShape (${s.length}) and rank of dy (${e.rank}) must match`);let a=s,i=e,o=!1;e.rank===4&&(o=!0,i=w(e,[1,e.shape[0],e.shape[1],e.shape[2],e.shape[3]]),a=[1,s[0],s[1],s[2],s[3]]);const u=a[4],p=i.shape[4];y(a.length===5,()=>`Error in conv3dDerInput: inShape must be length 5, but got length ${a.length}.`),y(i.rank===5,()=>`Error in conv3dDerInput: dy must be rank 5, but got rank ${i.rank}`),y(t.rank===5,()=>`Error in conv3dDerInput: filter must be rank 5, but got rank ${t.rank}`),y(u===t.shape[3],()=>`Error in conv3dDerInput: depth of input (${u}) must match input depth for filter ${t.shape[3]}.`),y(p===t.shape[4],()=>`Error in conv3dDerInput: depth of output (${p}) must match output depth for filter ${t.shape[4]}.`);const l={dy:i,filter:t},m={pad:r,strides:n,inputShape:a},d=b.runKernel(ar,l,m);return o?w(d,[d.shape[1],d.shape[2],d.shape[3],d.shape[4]]):d}const su=g({conv3DBackpropInput_:tu});function nu(s,e,t,n,r){const a=h(s,"x","conv3dTranspose"),i=h(e,"filter","conv3dTranspose");return su(t,a,i,n,r)}const ru=g({conv3dTranspose_:nu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function au(s){const t={x:h(s,"x","cos","float32")};return b.runKernel(or,t)}const ou=g({cos_:au});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function iu(s){const t={x:h(s,"x","cosh","float32")};return b.runKernel(ir,t)}const uu=g({cosh_:iu});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the 'License');
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an 'AS IS' BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cu(s,e=0,t=!1,n=!1){const a={x:h(s,"x","cumprod")},i={axis:e,exclusive:t,reverse:n};return b.runKernel(ur,a,i)}const pu=g({cumprod_:cu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lu(s,e=0,t=!1,n=!1){const a={x:h(s,"x","cumsum")},i={axis:e,exclusive:t,reverse:n};return b.runKernel(cr,a,i)}const mu=g({cumsum_:lu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hu(s,e,t,n=!1){const r=h(s,"x","denseBincount"),a=h(e,"weights","denseBincount");y(r.dtype==="int32",()=>`Error in denseBincount: input dtype must be int32, but got ${r.dtype}`),y(r.rank<=2,()=>`Error in denseBincount: input must be at most rank 2, but got rank ${r.rank}.`),y(t>=0,()=>`size must be non-negative, but got ${t}.`),y(a.size===r.size||a.size===0,()=>`Error in denseBincount: weights must have the same shape as x or 0-length, but got x shape: ${r.shape}, weights shape: ${a.shape}.`);const i={x:r,weights:a},o={size:t,binaryOutput:n};return b.runKernel(pr,i,o)}const du=g({denseBincount_:hu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fu(s,e,t="NHWC"){const n=h(s,"x","depthToSpace","float32"),r=t==="NHWC"?n.shape[1]:n.shape[2],a=t==="NHWC"?n.shape[2]:n.shape[3],i=t==="NHWC"?n.shape[3]:n.shape[1];y(e>1,()=>`blockSize should be > 1 for depthToSpace, but was: ${e}`),y(r*e>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${r} and ${e}  for depthToSpace with input shape
    ${n.shape}`),y(a*e>=0,()=>`Negative dimension size caused by overflow when multiplying
    ${a} and ${e} for depthToSpace with input shape
        ${n.shape}`),y(i%(e*e)===0,()=>`Dimension size must be evenly divisible by ${e*e} but is ${i} for depthToSpace with input shape ${n.shape}`);const o={x:n},u={blockSize:e,dataFormat:t};return b.runKernel(lr,o,u)}const yu=g({depthToSpace_:fu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gu(s,e,t,n,r="NHWC",a=[1,1],i){const o=h(s,"x","depthwiseConv2d","float32"),u=h(e,"filter","depthwiseConv2d","float32");let p=o,l=!1;o.rank===3&&(l=!0,p=w(o,[1,o.shape[0],o.shape[1],o.shape[2]])),y(p.rank===4,()=>`Error in depthwiseConv2d: input must be rank 4, but got rank ${p.rank}.`),y(u.rank===4,()=>`Error in depthwiseConv2d: filter must be rank 4, but got rank ${u.rank}.`);const m=r==="NHWC"?p.shape[3]:p.shape[1];y(m===u.shape[2],()=>`Error in depthwiseConv2d: number of input channels (${m}) must match the inChannels dimension in filter ${u.shape[2]}.`),oe("depthwiseConv2d",n,i);const d={x:p,filter:u},f={strides:t,pad:n,dataFormat:r,dilations:a,dimRoundingMode:i},N=b.runKernel(mr,d,f);return l?w(N,[N.shape[1],N.shape[2],N.shape[3]]):N}const ss=g({depthwiseConv2d_:gu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bu(s){const t={x:h(s,"x","diag")};return b.runKernel(hr,t)}const Nu=g({diag_:bu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tu(s,e,t,n,r=[1,1],a="NHWC"){const i=h(s,"x","dilation2d"),o=h(e,"filter","dilation2d");y(i.rank===3||i.rank===4,()=>`Error in dilation2d: input must be rank 3 or 4, but got rank ${i.rank}.`),y(o.rank===3,()=>`Error in dilation2d: filter must be rank 3, but got rank ${o.rank}.`),y(a==="NHWC",()=>`Error in dilation2d: Only NHWC is currently supported, but got dataFormat of ${a}`);let u=i,p=!1;i.rank===3&&(u=w(i,[1,i.shape[0],i.shape[1],i.shape[2]]),p=!0),y(u.shape[3]===o.shape[2],()=>`Error in dilation2d:  input and filter must have the same depth: ${u.shape[3]} vs ${o.shape[2]}`);const l={x:u,filter:o},m={strides:t,pad:n,dilations:r},d=b.runKernel(dr,l,m);return p?w(d,[d.shape[1],d.shape[2],d.shape[3]]):d}const wu=g({dilation2d_:Tu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $u(s,e){let t=h(s,"a","equal","string_or_numeric"),n=h(e,"b","equal","string_or_numeric");[t,n]=Q(t,n),G(t.shape,n.shape);const r={a:t,b:n};return b.runKernel(fr,r)}const Js=g({equal_:$u});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Su(s,e,t){const n=h(e,"a","where"),r=h(t,"b","where"),a=h(s,"condition","where","bool"),i=G(G(a.shape,n.shape),r.shape),o=tt(a,i),u=tt(n,i),p=tt(r,i),l={condition:o,t:u,e:p};return b.runKernel(yr,l)}const ge=g({where_:Su});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _u(s,e){let t=h(s,"a","div"),n=h(e,"b","div");[t,n]=Q(t,n);const r=X(t,n),a=Xt(r),i=Js(n,a);return ge(i,a,r)}const vu=g({divNoNan_:_u});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ku(s,e){const t=h(s,"t1","dot"),n=h(e,"t2","dot");y((t.rank===1||t.rank===2)&&(n.rank===1||n.rank===2),()=>`Error in dot: inputs must all be rank 1 or 2, but got ranks ${t.rank} and ${n.rank}.`);const r=t.rank===1?t.size:t.shape[1],a=n.rank===1?n.size:n.shape[0];if(y(r===a,()=>`Error in dot: inner dimensions of inputs must match, but got ${r} and ${a}.`),t.rank===1&&n.rank===1){const i=w(t,[1,-1]),o=w(n,[-1,1]),u=V(i,o);return w(u,[])}else if(t.rank===1&&n.rank===2){const i=w(t,[1,-1]),o=w(n,[n.shape[0],n.shape[1]]),u=V(i,o);return w(u,[u.size])}else if(t.rank===2&&n.rank===1){const i=w(n,[-1,1]),o=V(t,i);return w(o,[o.size])}else{const i=w(n,[n.shape[0],n.shape[1]]);return V(t,i)}}const Eu=g({dot_:ku});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xu(s,...e){const t=e.map((r,a)=>h(r,`tensors${a}`,"einsum")),n={equation:s};return b.runKernel(gr,t,n)}const Ae=g({einsum_:xu});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ou(s,e){const t=h(s,"x","ensureShape","string_or_numeric");if(!br(t.shape,e))throw new Error(`EnsureShape: Shape of tensor ${t.shape} is not compatible with expected shape ${e}`);return s}const Iu=g({ensureShape_:Ou});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Du(s){let e=h(s,"x","erf");y(e.dtype==="int32"||e.dtype==="float32",()=>"Input dtype must be `int32` or `float32`."),e.dtype==="int32"&&(e=K(e,"float32"));const t={x:e};return b.runKernel(Nr,t)}const Au=g({erf_:Du});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cu(s,e=null,t=!1){const r={x:h(s,"x","max")},a={reductionIndices:e,keepDims:t};return b.runKernel(Tr,r,a)}const ze=g({max_:Cu});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zu(s,e=null,t=!1){const r={x:h(s,"x","min")},a={axis:e,keepDims:t};return b.runKernel(wr,r,a)}const zt=g({min_:zu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pu(s,e="euclidean",t=null,n=!1){s=h(s,"x","norm");const r=Ms(s,e,t);let a=r.shape;if(n){const i=Yt(t,s.shape);a=Tt(r.shape,i)}return w(r,a)}function Ms(s,e,t=null){if(s.rank===0)return se(s);if(s.rank!==1&&t===null)return Ms(w(s,[-1]),e,t);if(s.rank===1||typeof t=="number"||Array.isArray(t)&&t.length===1){if(e===1)return R(se(s),t);if(e===1/0)return ze(se(s),t);if(e===-1/0)return zt(se(s),t);if(e==="euclidean"||e===2)return At(R(Qt(se(s),L(2,"int32")),t));throw new Error(`Error in norm: invalid ord value: ${e}`)}if(Array.isArray(t)&&t.length===2){if(e===1)return ze(R(se(s),t[0]),t[1]-1);if(e===1/0)return ze(R(se(s),t[1]),t[0]);if(e===-1/0)return zt(R(se(s),t[1]),t[0]);if(e==="fro"||e==="euclidean")return At(R(wt(s),t));throw new Error(`Error in norm: invalid ord value: ${e}`)}throw new Error(`Error in norm: invalid axis: ${t}`)}const vt=g({norm_:Pu});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vu(s,e=null,t=!1){return vt(s,"euclidean",e,t)}const Lu=g({euclideanNorm_:Vu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fu(s){const t={x:h(s,"x","exp")};return b.runKernel($r,t)}const ke=g({exp_:Fu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ru(s,e=0){const t=h(s,"x","expandDims","string_or_numeric");y(e<=t.rank,()=>"Axis must be <= rank of the tensor");const n={input:t},r={dim:e};return b.runKernel(Sr,n,r)}const ye=g({expandDims_:Ru});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bu(s){const t={x:h(s,"x","expm1")};return b.runKernel(_r,t)}const ju=g({expm1_:Bu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qu(s,e){const t=h(s,"x","tile","string_or_numeric");y(t.rank===e.length,()=>`Error in transpose: rank of input ${t.rank} must match length of reps ${e}.`);const n={x:t},r={reps:e};return b.runKernel(vr,n,r)}const He=g({tile_:qu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ku(s,e,t,n="float32"){e==null&&(e=s);const r=Fe([s,e],n),a=s<=e?s:e;for(let o=0;o<a;++o)r.set(1,o,o);const i=w(r.toTensor(),[s,e]);if(t==null)return i;if(t.length===1)return He(ye(i,0),[t[0],1,1]);if(t.length===2)return He(ye(ye(i,0),0),[t[0],t[1],1,1]);if(t.length===3)return He(ye(ye(ye(i,0),0),0),[t[0],t[1],t[2],1,1]);throw new Error(`eye() currently supports only 1D and 2D batchShapes, but received ${t.length}D.`)}const en=g({eye_:Ku});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Wu(s){const t={x:h(s,"x","floor","float32")};return b.runKernel(kr,t)}const tn=g({floor_:Wu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hu(s,e,t=0,n=0){const r=h(s,"x","gather"),a=h(e,"indices","gather","int32"),i={x:r,indices:a},o={axis:t,batchDims:n};return b.runKernel(Er,i,o)}const sn=g({gather_:Hu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gu(s,e){let t=h(s,"a","greater","string_or_numeric"),n=h(e,"b","greater","string_or_numeric");[t,n]=Q(t,n),G(t.shape,n.shape);const r={a:t,b:n};return b.runKernel(xr,r)}const kt=g({greater_:Gu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uu(s,e){let t=h(s,"a","greaterEqual","string_or_numeric"),n=h(e,"b","greaterEqual","string_or_numeric");[t,n]=Q(t,n),G(t.shape,n.shape);const r={a:t,b:n};return b.runKernel(Or,r)}const nn=g({greaterEqual_:Uu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xu(s){const t={input:h(s,"input","imag")};return b.runKernel(Ir,t)}const Et=g({imag_:Xu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yu(s){const t={x:h(s,"x","isFinite")};return b.runKernel(Dr,t)}const Qu=g({isFinite_:Yu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zu(s){const t={x:h(s,"x","isInf")};return b.runKernel(Ar,t)}const Ju=g({isInf_:Zu});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mu(s){const t={x:h(s,"x","isNaN")};return b.runKernel(Cr,t)}const ec=g({isNaN_:Mu});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tc(s,e){let t=h(s,"a","less","string_or_numeric"),n=h(e,"b","less","string_or_numeric");[t,n]=Q(t,n),G(t.shape,n.shape);const r={a:t,b:n};return b.runKernel(zr,r)}const Pt=g({less_:tc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sc(s,e){let t=h(s,"a","lessEqual","string_or_numeric"),n=h(e,"b","lessEqual","string_or_numeric");[t,n]=Q(t,n),G(t.shape,n.shape);const r={a:t,b:n};return b.runKernel(Pr,r)}const ns=g({lessEqual_:sc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nc(s,e,t){if(t<=0)throw new Error("The number of values should be positive.");const n={start:s,stop:e,num:t};return b.runKernel(Vr,{},n)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rc(s,e=5,t=1,n=1,r=.5){const a=h(s,"x","localResponseNormalization");y(a.rank===4||a.rank===3,()=>`Error in localResponseNormalization: x must be rank 3 or 4 but got
               rank ${a.rank}.`),y(Fs(e),()=>`Error in localResponseNormalization: depthRadius must be an integer but got depthRadius ${e}.`);let i=a,o=!1;a.rank===3&&(o=!0,i=w(a,[1,a.shape[0],a.shape[1],a.shape[2]]));const u={x:i},p={depthRadius:e,bias:t,alpha:n,beta:r},l=b.runKernel(Lr,u,p);return o?w(l,[l.shape[1],l.shape[2],l.shape[3]]):l}const ac=g({localResponseNormalization_:rc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function oc(s){const t={x:h(s,"x","log","float32")};return b.runKernel(Fr,t)}const Xe=g({log_:oc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ic(s){const t={x:h(s,"x","log1p")};return b.runKernel(Rr,t)}const rn=g({log1p_:ic});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uc(s){const t={x:h(s,"x","neg")};return b.runKernel(Br,t)}const ue=g({neg_:uc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cc(s){const t={x:h(s,"x","softplus")};return b.runKernel(jr,t)}const an=g({softplus_:cc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function pc(s){const e=h(s,"x","logSigmoid");return he(n=>({value:ue(an(ue(n))),gradFunc:i=>I(i,We(ue(n)))}))(e)}const lc=g({logSigmoid_:pc});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function mc(s,e=-1){const t=h(s,"logits","logSoftmax");if(e===-1&&(e=t.rank-1),e!==t.rank-1)throw Error(`Log Softmax along a non-last dimension is not yet supported. Logits was rank ${t.rank} and axis was ${e}`);return he((r,a)=>{const o=ze(r,e,!0),u=P(r,o),p=P(K(u,"float32"),Xe(R(ke(u),e,!0)));return a([p]),{value:p,gradFunc:(m,d)=>{const[f]=d,N=!0,$=ke(f);return P(m,I(R(m,e,N),$))}}})(t)}const hc=g({logSoftmax_:mc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dc(s,e=null,t=!1){const n=h(s,"x","logSumExp"),r=Yt(e,n.shape),a=ze(n,r,!0),i=P(n,a),o=ke(i),u=R(o,r),p=Xe(u),l=W(w(a,p.shape),p);if(t){const m=Tt(l.shape,r);return w(l,m)}return l}const on=g({logSumExp_:dc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fc(s,e){const t=h(s,"a","logicalAnd","bool"),n=h(e,"b","logicalAnd","bool");G(t.shape,n.shape);const r={a:t,b:n};return b.runKernel(qr,r)}const mt=g({logicalAnd_:fc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yc(s){const t={x:h(s,"x","logicalNot","bool")};return b.runKernel(Kr,t)}const un=g({logicalNot_:yc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gc(s,e){const t=h(s,"a","logicalOr","bool"),n=h(e,"b","logicalOr","bool");G(t.shape,n.shape);const r={a:t,b:n};return b.runKernel(Wr,r)}const cn=g({logicalOr_:gc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bc(s,e){const t=h(s,"a","logicalXor","bool"),n=h(e,"b","logicalXor","bool");return G(t.shape,n.shape),mt(cn(s,e),un(mt(s,e)))}const Nc=g({logicalXor_:bc});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Me=2147483648;function Tc(s,e,t="left"){const n=h(s,"sortedSequence","searchSorted"),r=h(e,"values","searchSorted"),a=n.shape[n.shape.length-1],i=r.shape[r.shape.length-1],o=w(n,[-1,a]),u=w(r,[-1,i]);if(o.rank<2)throw new Error("Sorted input argument must be at least 2-dimensional");if(o.shape[0]!==u.shape[0])throw new Error("Leading dimension of 'sortedSequence' and 'values' must match.");if(Se(u.shape)>=Me)throw new Error(`values tensor size must less than ${Me}`);if(o.shape[1]>=Me)throw new Error(`trailing dim_size must less than ${Me} for int32 output type, was ${o.shape[1]}`);const p={sortedSequence:o,values:u},l={side:t};return b.runKernel(Hr,p,l)}const rs=g({searchSorted_:Tc});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function wc(s,e){return rs(s,e,"left")}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $c(s,e,t,n,r){const a=h(s,"x","maxPool"),i=1;let o=a,u=!1;a.rank===3&&(u=!0,o=w(a,[1,a.shape[0],a.shape[1],a.shape[2]])),y(o.rank===4,()=>`Error in maxPool: input must be rank 4 but got rank ${o.rank}.`),y(be(t,i),()=>`Error in maxPool: Either strides or dilations must be 1. Got strides ${t} and dilations '${i}'`),oe("maxPool",n,r);const p={x:o},l={filterSize:e,strides:t,pad:n,dimRoundingMode:r},m=b.runKernel(Gr,p,l);return u?w(m,[m.shape[1],m.shape[2],m.shape[3]]):m}const pn=g({maxPool_:$c});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sc(s,e=[1,1,1],t,n,r,a="NDHWC"){const i=h(s,"x","maxPool3d");let o=i,u=!1;i.rank===4&&(u=!0,o=w(i,[1,i.shape[0],i.shape[1],i.shape[2],i.shape[3]])),y(o.rank===5,()=>`Error in maxPool3d: x must be rank 5 but got rank ${o.rank}.`),y(a==="NDHWC",()=>`Error in maxPool3d: Only NDHWC is currently supported, but got dataFormat of ${a}`),oe("maxPool3d",n,r);const p={x:o},l={filterSize:e,strides:t,pad:n,dimRoundingMode:r,dataFormat:a},m=b.runKernel(Ur,p,l);return u?w(m,[m.shape[1],m.shape[2],m.shape[3],m.shape[4]]):m}const _c=g({maxPool3d_:Sc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vc(s,e,t,n,r=!1){const i={x:h(s,"x","maxPoolWithArgmax")},o={filterSize:e,strides:t,pad:n,includeBatchInIndex:r},u=b.runKernel(Xr,i,o);return{result:u[0],indexes:u[1]}}const kc=g({maxPoolWithArgmax_:vc});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ec(s,e=null,t=!1){const r={x:h(s,"x","mean")},a={axis:e,keepDims:t};return b.runKernel(Yr,r,a)}const ht=g({mean_:Ec});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $e(s,e="float32"){if(Ne(s),e==="complex64"){const n=$e(s,"float32"),r=$t(s,"float32");return Le(n,r)}const t=Qr(Se(s),e);return b.makeTensor(t,s,e)}/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xc(s,e,{indexing:t="xy"}={}){if(t!=="xy"&&t!=="ij")throw new TypeError(`${t} is not a valid third argument to meshgrid`);if(s===void 0)return[];let n=h(s,"x","meshgrid",s instanceof _e?s.dtype:"float32");if(e===void 0)return[n];let r=h(e,"y","meshgrid",e instanceof _e?e.dtype:"float32");const a=Se(n.shape),i=Se(r.shape);return t==="xy"?(n=w(n,[1,-1]),r=w(r,[-1,1]),[V($e([i,1],n.dtype),n),V(r,$e([1,a],r.dtype))]):(n=w(n,[-1,1]),r=w(r,[1,-1]),[V(n,$e([1,i],n.dtype)),V($e([a,1],r.dtype),r)])}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oc(s,e){let t=h(s,"a","minimum"),n=h(e,"b","minimum");[t,n]=Q(t,n),t.dtype==="bool"&&(t=K(t,"int32"),n=K(n,"int32")),G(t.shape,n.shape);const r={a:t,b:n};return b.runKernel(Zr,r)}const dt=g({minimum_:Oc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ic(s,e,t){y(t==="reflect"||t==="symmetric",()=>`Invalid mode. Mode must be either reflect or symmetric. Got ${t}.`);const n=h(s,"x","mirrorPad");if(n.rank===0)throw new Error("mirrorPad(scalar) is not defined. Pass non-scalar to mirrorPad");y(e.length===n.rank,()=>`Padding doesn't match input. Must be ${n.rank}. Got ${e.length}.`);const r=t==="reflect"?1:0;for(let o=0;o<n.rank;o++)y(e[o].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),y(e[o][0]>=0&&e[o][0]<=n.shape[o]-r&&e[o][1]>=0&&e[o][1]<=n.shape[o]-r,()=>`Padding in dimension ${o} cannot be greater than or equal to ${n.shape[o]-r} or less than 0 for input of shape ${n.shape}`);const a={paddings:e,mode:t},i={x:n};return b.runKernel(Jr,i,a)}const Dc=g({mirrorPad_:Ic});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ac(s,e){let t=h(s,"a","mod"),n=h(e,"b","mod");[t,n]=Q(t,n);const r={a:t,b:n};return b.runKernel(Mr,r)}const Cc=g({mod_:Ac});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zc(s,e=null,t=!1){s=h(s,"x","moments");const n=Yt(e,s.shape),r=ht(s,n,t);let a=r.shape;t||(a=Tt(r.shape,n));const i=wt(P(K(s,"float32"),w(r,a))),o=ht(i,n,t);return{mean:r,variance:o}}const Pc=g({moments_:zc});function Vc(s,e,t,n){const r=h(e,"data","multiRNNCell"),a=pt(t,"c","multiRNNCell"),i=pt(n,"h","multiRNNCell");let o=r;const u=[];for(let m=0;m<s.length;m++){const d=s[m](o,a[m],i[m]);u.push(d[0]),u.push(d[1]),o=d[1]}const p=[],l=[];for(let m=0;m<u.length;m+=2)p.push(u[m]),l.push(u[m+1]);return[p,l]}const Lc=g({multiRNNCell_:Vc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fc(s,e,t,n=!1){const r=h(s,"logits","multinomial"),a=r.size,i=r.rank;if(a<2)throw new Error(`Error in multinomial: you need at least 2 outcomes, but got ${a}.`);if(i>2)throw new Error(`Rank of probabilities must be 1 or 2, but is ${i}`);t=t||Math.random();const u={logits:i===1?w(r,[1,-1]):r},p={numSamples:e,seed:t,normalized:n},l=b.runKernel(ea,u,p);return i===1?w(l,[l.size]):l}const Rc=g({multinomial_:Fc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bc(s,e){let t=h(s,"a","notEqual","string_or_numeric"),n=h(e,"b","notEqual","string_or_numeric");[t,n]=Q(t,n),G(t.shape,n.shape);const r={a:t,b:n};return b.runKernel(ta,r)}const ln=g({notEqual_:Bc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jc(s,e,t=1,n=0,r="int32"){if(e<2)throw new Error(`Error in oneHot: depth must be >=2, but it is ${e}`);const i={indices:h(s,"indices","oneHot","int32")},o={dtype:r,depth:e,onValue:t,offValue:n};return b.runKernel(sa,i,o)}const qc=g({oneHot_:jc});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kc(s){const t={x:h(s,"x","onesLike")};return b.runKernel(na,t)}const Wc=g({onesLike_:Kc});function Hc(s,e){const t=h(s,"v1","outerProduct"),n=h(e,"v2","outerProduct");y(t.rank===1&&n.rank===1,()=>`Error in outerProduct: inputs must be rank 1, but got ranks ${t.rank} and ${n.rank}.`);const r=w(t,[-1,1]),a=w(n,[1,-1]);return V(r,a)}const Gc=g({outerProduct_:Hc});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uc(s,e,t=0){const n=h(s,"x","pad");if(n.rank===0)throw new Error("pad(scalar) is not defined. Pass non-scalar to pad");const r={paddings:e,constantValue:t},a={x:n};return b.runKernel(ra,a,r)}const Je=g({pad_:Uc});function Xc(s,e,t=0){return y(e.length===2,()=>"Invalid number of paddings. Must be length of 2."),Je(s,[e],t)}const Yc=g({pad1d_:Xc});function Qc(s,e,t=0){return y(e.length===2&&e[0].length===2&&e[1].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Je(s,e,t)}const Zc=g({pad2d_:Qc});function Jc(s,e,t=0){return y(e.length===3&&e[0].length===2&&e[1].length===2&&e[2].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Je(s,e,t)}const Mc=g({pad3d_:Jc});function ep(s,e,t=0){return y(e.length===4&&e[0].length===2&&e[1].length===2&&e[2].length===2&&e[3].length===2,()=>"Invalid number of paddings. Must be length of 2 each."),Je(s,e,t)}const tp=g({pad4d_:ep});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function sp(s,e,t){const n=h(s,"x","spaceToBatchND");y(n.rank>=1+e.length,()=>`input rank ${n.rank} should be > than [blockShape] ${e.length}`),y(t.length===e.length,()=>`paddings.shape[0] ${t.length} must be equal to [blockShape] ${e.length}`),y(n.shape.reduce((i,o,u)=>u>0&&u<=e.length?i&&(o+t[u-1][0]+t[u-1][1])%e[u-1]===0:i,!0),()=>`input spatial dimensions ${n.shape.slice(1)} with paddings ${t.toString()} must be divisible by blockShapes ${e.toString()}`);const r={x:n},a={blockShape:e,paddings:t};return b.runKernel(aa,r,a)}const mn=g({spaceToBatchND_:sp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function np(s,e,t,n,r,a,i){r==null&&(r=[1,1]),a==null&&(a=1),n===0&&(n="valid");const o=h(s,"x","maxPool");let u=o,p=!1;o.rank===3&&(p=!0,u=w(o,[1,o.shape[0],o.shape[1],o.shape[2]])),y(be(a,r),()=>`Error in pool: Either strides or dilations must be 1. Got strides ${a} and dilations '${r}'`);const l=oa(u.shape,e,a,r,n),m=[l.dilationHeight,l.dilationWidth];let d;n==="same"?d=ap([l.filterHeight,l.filterWidth],m):d=[[0,0],[0,0]];const f=m[0]===1&&m[1]===1,[N,$]=rp([l.inHeight,l.inWidth],m,d),T=f?n:"valid",S=f?u:mn(u,m,N),C=(t==="avg"?()=>Xs(S,e,a,T,i):()=>pn(S,e,a,T,i))(),_=f?C:Ys(C,m,$);return p?w(_,[_.shape[1],_.shape[2],_.shape[3]]):_}function rp(s,e,t){const n=t.map(l=>l[0]),r=t.map(l=>l[1]),a=s.concat(n,r),i=e.map((l,m)=>(l-a[m]%l)%l),o=r.map((l,m)=>l+i[m]),u=e.map((l,m)=>[n[m],o[m]]),p=e.map((l,m)=>[0,i[m]]);return[u,p]}function ap(s,e){const n=s.map((i,o)=>i+(i-1)*(e[o]-1)).map(i=>i-1),r=n.map(i=>Math.floor(i/2)),a=n.map((i,o)=>i-r[o]);return n.map((i,o)=>[r[o],a[o]])}const op=g({pool_:np});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ip(s,e=null,t=!1){let n=h(s,"x","prod");n.dtype==="bool"&&(n=K(n,"int32"));const r={x:n},a={axis:e,keepDims:t};return b.runKernel(ia,r,a)}const up=g({prod_:ip});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function cp(s,e,t,n){const r=s.map((l,m)=>h(l,`tensors${m}`,"raggedGather","int32")),a=h(e,"paramsDenseValues","raggedGather"),i=h(t,"indices","raggedGather","int32"),o={paramsNestedSplits:r,paramsDenseValues:a,indices:i},u={outputRaggedRank:n},p=b.runKernel(ua,o,u);return{outputNestedSplits:p.slice(0,p.length-1),outputDenseValues:p[p.length-1]}}const pp=g({raggedGather_:cp});/**
 * @license
 * Copyright 2022 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lp(s,e,t){const n=h(s,"starts","raggedRange"),r=h(e,"limits","raggedRange",n.dtype),a=h(t,"deltas","raggedRange",n.dtype),i={starts:n,limits:r,deltas:a},o=b.runKernel(ca,i);return{rtNestedSplits:o[0],rtDenseValues:o[1]}}const mp=g({raggedRange_:lp});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hp(s,e,t,n,r){const a=h(s,"shape","raggedTensorToTensor","int32"),i=h(e,"values","raggedTensorToTensor"),o=h(t,"defaultValue","raggedTensorToTensor",i.dtype),u=n.map((m,d)=>h(m,`tensors${d}`,"raggedTensorToTensor","int32")),p={shape:a,values:i,defaultValue:o,rowPartitionTensors:u},l={rowPartitionTypes:r};return b.runKernel(pa,p,l)}const dp=g({raggedTensorToTensor_:hp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fp(s,e,t){Ne(s);const n=Se(s);let r=null;if(t==null||t==="float32")r=new Float32Array(n);else if(t==="int32")r=new Int32Array(n);else if(t==="bool")r=new Uint8Array(n);else throw new Error(`Unknown data type ${t}`);for(let a=0;a<n;a++)r[a]=e();return b.makeTensor(r,s,t)}const yp=g({rand_:fp});var st={exports:{}},gp=st.exports,Ns;function bp(){return Ns||(Ns=1,(function(s){(function(e,t,n){function r(u){var p=this,l=o();p.next=function(){var m=2091639*p.s0+p.c*23283064365386963e-26;return p.s0=p.s1,p.s1=p.s2,p.s2=m-(p.c=m|0)},p.c=1,p.s0=l(" "),p.s1=l(" "),p.s2=l(" "),p.s0-=l(u),p.s0<0&&(p.s0+=1),p.s1-=l(u),p.s1<0&&(p.s1+=1),p.s2-=l(u),p.s2<0&&(p.s2+=1),l=null}function a(u,p){return p.c=u.c,p.s0=u.s0,p.s1=u.s1,p.s2=u.s2,p}function i(u,p){var l=new r(u),m=p&&p.state,d=l.next;return d.int32=function(){return l.next()*4294967296|0},d.double=function(){return d()+(d()*2097152|0)*11102230246251565e-32},d.quick=d,m&&(typeof m=="object"&&a(m,l),d.state=function(){return a(l,{})}),d}function o(){var u=4022871197,p=function(l){l=String(l);for(var m=0;m<l.length;m++){u+=l.charCodeAt(m);var d=.02519603282416938*u;u=d>>>0,d-=u,d*=u,u=d>>>0,d-=u,u+=d*4294967296}return(u>>>0)*23283064365386963e-26};return p}t&&t.exports?t.exports=i:this.alea=i})(gp,s)})(st)),st.exports}var nt={exports:{}},Np=nt.exports,Ts;function Tp(){return Ts||(Ts=1,(function(s){(function(e,t,n){function r(o){var u=this,p="";u.x=0,u.y=0,u.z=0,u.w=0,u.next=function(){var m=u.x^u.x<<11;return u.x=u.y,u.y=u.z,u.z=u.w,u.w^=u.w>>>19^m^m>>>8},o===(o|0)?u.x=o:p+=o;for(var l=0;l<p.length+64;l++)u.x^=p.charCodeAt(l)|0,u.next()}function a(o,u){return u.x=o.x,u.y=o.y,u.z=o.z,u.w=o.w,u}function i(o,u){var p=new r(o),l=u&&u.state,m=function(){return(p.next()>>>0)/4294967296};return m.double=function(){do var d=p.next()>>>11,f=(p.next()>>>0)/4294967296,N=(d+f)/(1<<21);while(N===0);return N},m.int32=p.next,m.quick=m,l&&(typeof l=="object"&&a(l,p),m.state=function(){return a(p,{})}),m}t&&t.exports?t.exports=i:this.xor128=i})(Np,s)})(nt)),nt.exports}var rt={exports:{}},wp=rt.exports,ws;function $p(){return ws||(ws=1,(function(s){(function(e,t,n){function r(o){var u=this,p="";u.next=function(){var m=u.x^u.x>>>2;return u.x=u.y,u.y=u.z,u.z=u.w,u.w=u.v,(u.d=u.d+362437|0)+(u.v=u.v^u.v<<4^(m^m<<1))|0},u.x=0,u.y=0,u.z=0,u.w=0,u.v=0,o===(o|0)?u.x=o:p+=o;for(var l=0;l<p.length+64;l++)u.x^=p.charCodeAt(l)|0,l==p.length&&(u.d=u.x<<10^u.x>>>4),u.next()}function a(o,u){return u.x=o.x,u.y=o.y,u.z=o.z,u.w=o.w,u.v=o.v,u.d=o.d,u}function i(o,u){var p=new r(o),l=u&&u.state,m=function(){return(p.next()>>>0)/4294967296};return m.double=function(){do var d=p.next()>>>11,f=(p.next()>>>0)/4294967296,N=(d+f)/(1<<21);while(N===0);return N},m.int32=p.next,m.quick=m,l&&(typeof l=="object"&&a(l,p),m.state=function(){return a(p,{})}),m}t&&t.exports?t.exports=i:this.xorwow=i})(wp,s)})(rt)),rt.exports}var at={exports:{}},Sp=at.exports,$s;function _p(){return $s||($s=1,(function(s){(function(e,t,n){function r(o){var u=this;u.next=function(){var l=u.x,m=u.i,d,f;return d=l[m],d^=d>>>7,f=d^d<<24,d=l[m+1&7],f^=d^d>>>10,d=l[m+3&7],f^=d^d>>>3,d=l[m+4&7],f^=d^d<<7,d=l[m+7&7],d=d^d<<13,f^=d^d<<9,l[m]=f,u.i=m+1&7,f};function p(l,m){var d,f=[];if(m===(m|0))f[0]=m;else for(m=""+m,d=0;d<m.length;++d)f[d&7]=f[d&7]<<15^m.charCodeAt(d)+f[d+1&7]<<13;for(;f.length<8;)f.push(0);for(d=0;d<8&&f[d]===0;++d);for(d==8?f[7]=-1:f[d],l.x=f,l.i=0,d=256;d>0;--d)l.next()}p(u,o)}function a(o,u){return u.x=o.x.slice(),u.i=o.i,u}function i(o,u){o==null&&(o=+new Date);var p=new r(o),l=u&&u.state,m=function(){return(p.next()>>>0)/4294967296};return m.double=function(){do var d=p.next()>>>11,f=(p.next()>>>0)/4294967296,N=(d+f)/(1<<21);while(N===0);return N},m.int32=p.next,m.quick=m,l&&(l.x&&a(l,p),m.state=function(){return a(p,{})}),m}t&&t.exports?t.exports=i:this.xorshift7=i})(Sp,s)})(at)),at.exports}var ot={exports:{}},vp=ot.exports,Ss;function kp(){return Ss||(Ss=1,(function(s){(function(e,t,n){function r(o){var u=this;u.next=function(){var l=u.w,m=u.X,d=u.i,f,N;return u.w=l=l+1640531527|0,N=m[d+34&127],f=m[d=d+1&127],N^=N<<13,f^=f<<17,N^=N>>>15,f^=f>>>12,N=m[d]=N^f,u.i=d,N+(l^l>>>16)|0};function p(l,m){var d,f,N,$,T,S=[],O=128;for(m===(m|0)?(f=m,m=null):(m=m+"\0",f=0,O=Math.max(O,m.length)),N=0,$=-32;$<O;++$)m&&(f^=m.charCodeAt(($+32)%m.length)),$===0&&(T=f),f^=f<<10,f^=f>>>15,f^=f<<4,f^=f>>>13,$>=0&&(T=T+1640531527|0,d=S[$&127]^=f+T,N=d==0?N+1:0);for(N>=128&&(S[(m&&m.length||0)&127]=-1),N=127,$=512;$>0;--$)f=S[N+34&127],d=S[N=N+1&127],f^=f<<13,d^=d<<17,f^=f>>>15,d^=d>>>12,S[N]=f^d;l.w=T,l.X=S,l.i=N}p(u,o)}function a(o,u){return u.i=o.i,u.w=o.w,u.X=o.X.slice(),u}function i(o,u){o==null&&(o=+new Date);var p=new r(o),l=u&&u.state,m=function(){return(p.next()>>>0)/4294967296};return m.double=function(){do var d=p.next()>>>11,f=(p.next()>>>0)/4294967296,N=(d+f)/(1<<21);while(N===0);return N},m.int32=p.next,m.quick=m,l&&(l.X&&a(l,p),m.state=function(){return a(p,{})}),m}t&&t.exports?t.exports=i:this.xor4096=i})(vp,s)})(ot)),ot.exports}var it={exports:{}},Ep=it.exports,_s;function xp(){return _s||(_s=1,(function(s){(function(e,t,n){function r(o){var u=this,p="";u.next=function(){var m=u.b,d=u.c,f=u.d,N=u.a;return m=m<<25^m>>>7^d,d=d-f|0,f=f<<24^f>>>8^N,N=N-m|0,u.b=m=m<<20^m>>>12^d,u.c=d=d-f|0,u.d=f<<16^d>>>16^N,u.a=N-m|0},u.a=0,u.b=0,u.c=-1640531527,u.d=1367130551,o===Math.floor(o)?(u.a=o/4294967296|0,u.b=o|0):p+=o;for(var l=0;l<p.length+20;l++)u.b^=p.charCodeAt(l)|0,u.next()}function a(o,u){return u.a=o.a,u.b=o.b,u.c=o.c,u.d=o.d,u}function i(o,u){var p=new r(o),l=u&&u.state,m=function(){return(p.next()>>>0)/4294967296};return m.double=function(){do var d=p.next()>>>11,f=(p.next()>>>0)/4294967296,N=(d+f)/(1<<21);while(N===0);return N},m.int32=p.next,m.quick=m,l&&(typeof l=="object"&&a(l,p),m.state=function(){return a(p,{})}),m}t&&t.exports?t.exports=i:this.tychei=i})(Ep,s)})(it)),it.exports}var ut={exports:{}};const Op={},Ip=Object.freeze(Object.defineProperty({__proto__:null,default:Op},Symbol.toStringTag,{value:"Module"})),Dp=jo(Ip);var Ap=ut.exports,vs;function Cp(){return vs||(vs=1,(function(s){(function(e,t,n){var r=256,a=6,i=52,o="random",u=n.pow(r,a),p=n.pow(2,i),l=p*2,m=r-1,d;function f(_,k,v){var E=[];k=k==!0?{entropy:!0}:k||{};var x=S(T(k.entropy?[_,C(t)]:_??O(),3),E),D=new N(E),A=function(){for(var z=D.g(a),j=u,B=0;z<p;)z=(z+B)*r,j*=r,B=D.g(1);for(;z>=l;)z/=2,j/=2,B>>>=1;return(z+B)/j};return A.int32=function(){return D.g(4)|0},A.quick=function(){return D.g(4)/4294967296},A.double=A,S(C(D.S),t),(k.pass||v||function(z,j,B,q){return q&&(q.S&&$(q,D),z.state=function(){return $(D,{})}),B?(n[o]=z,j):z})(A,x,"global"in k?k.global:this==n,k.state)}function N(_){var k,v=_.length,E=this,x=0,D=E.i=E.j=0,A=E.S=[];for(v||(_=[v++]);x<r;)A[x]=x++;for(x=0;x<r;x++)A[x]=A[D=m&D+_[x%v]+(k=A[x])],A[D]=k;(E.g=function(z){for(var j,B=0,q=E.i,Z=E.j,te=E.S;z--;)j=te[q=m&q+1],B=B*r+te[m&(te[q]=te[Z=m&Z+j])+(te[Z]=j)];return E.i=q,E.j=Z,B})(r)}function $(_,k){return k.i=_.i,k.j=_.j,k.S=_.S.slice(),k}function T(_,k){var v=[],E=typeof _,x;if(k&&E=="object")for(x in _)try{v.push(T(_[x],k-1))}catch{}return v.length?v:E=="string"?_:_+"\0"}function S(_,k){for(var v=_+"",E,x=0;x<v.length;)k[m&x]=m&(E^=k[m&x]*19)+v.charCodeAt(x++);return C(k)}function O(){try{var _;return d&&(_=d.randomBytes)?_=_(r):(_=new Uint8Array(r),(e.crypto||e.msCrypto).getRandomValues(_)),C(_)}catch{var k=e.navigator,v=k&&k.plugins;return[+new Date,e,v,e.screen,C(t)]}}function C(_){return String.fromCharCode.apply(0,_)}if(S(n.random(),t),s.exports){s.exports=f;try{d=Dp}catch{}}else n["seed"+o]=f})(typeof self<"u"?self:Ap,[],Math)})(ut)),ut.exports}var Ot,ks;function zp(){if(ks)return Ot;ks=1;var s=bp(),e=Tp(),t=$p(),n=_p(),r=kp(),a=xp(),i=Cp();return i.alea=s,i.xor128=e,i.xorwow=t,i.xorshift7=n,i.xor4096=r,i.tychei=a,Ot=i,Ot}var as=zp();/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class os{constructor(e,t,n,r,a){this.mean=e,this.stdDev=t,this.dtype=n,this.nextVal=NaN,this.truncated=r,this.truncated&&(this.upper=this.mean+this.stdDev*2,this.lower=this.mean-this.stdDev*2);const i=a||Math.random();this.random=as.alea(i.toString())}nextValue(){if(!isNaN(this.nextVal)){const r=this.nextVal;return this.nextVal=NaN,r}let e,t,n=!1;for(;!n;){let r,a,i;do r=2*this.random()-1,a=2*this.random()-1,i=r*r+a*a;while(i>=1||i===0);const o=Math.sqrt(-2*Math.log(i)/i);e=this.mean+this.stdDev*r*o,t=this.mean+this.stdDev*a*o,(!this.truncated||this.isValidTruncated(e))&&(n=!0)}return(!this.truncated||this.isValidTruncated(t))&&(this.nextVal=this.convertValue(t)),this.convertValue(e)}convertValue(e){return this.dtype==null||this.dtype==="float32"?e:Math.round(e)}isValidTruncated(e){return e<=this.upper&&e>=this.lower}}class Pp{constructor(e,t,n,r){this.alpha=e,this.beta=1/t,this.dtype=n;const a=r||Math.random();this.randu=as.alea(a.toString()),this.randn=new os(0,1,n,!1,this.randu()),e<1?this.d=e+2/3:this.d=e-1/3,this.c=1/Math.sqrt(9*this.d)}nextValue(){let e,t,n,r,a,i;for(;;){do r=this.randn.nextValue(),i=1+this.c*r;while(i<=0);if(i*=i*i,e=r*r,t=1-.331*e*e,n=.5*e+this.d*(1-i+Math.log(i)),a=this.randu(),a<t||Math.log(a)<n)break}return i=1/this.beta*this.d*i,this.alpha<1&&(i*=Math.pow(this.randu(),1/this.alpha)),this.convertValue(i)}convertValue(e){return this.dtype==="float32"?e:Math.round(e)}}class Vp{constructor(e=0,t=1,n,r){if(this.canReturnFloat=()=>this.dtype==null||this.dtype==="float32",this.min=e,this.range=t-e,this.dtype=n,r==null&&(r=Math.random()),typeof r=="number"&&(r=r.toString()),!this.canReturnFloat()&&this.range<=1)throw new Error(`The difference between ${e} - ${t} <= 1 and dtype is not float`);this.random=as.alea(r)}convertValue(e){return this.canReturnFloat()?e:Math.round(e)}nextValue(){return this.convertValue(this.min+this.range*this.random())}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lp(s,e,t=1,n="float32",r){if(Ne(s),t==null&&(t=1),n==null&&(n="float32"),n!=="float32"&&n!=="int32")throw new Error(`Unsupported data type ${n}`);const a=new Pp(e,t,n,r),i=Fe(s,n);for(let o=0;o<i.values.length;o++)i.values[o]=a.nextValue();return i.toTensor()}const Fp=g({randomGamma_:Lp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rp(s,e=0,t=1,n,r){if(Ne(s),n!=null&&n==="bool")throw new Error(`Unsupported data type ${n}`);const a=new os(e,t,n,!1,r),i=Fe(s,n);for(let o=0;o<i.values.length;o++)i.values[o]=a.nextValue();return i.toTensor()}const hn=g({randomNormal_:Rp});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bp(s,e,t){if(e!=null&&e==="bool")throw new Error(`Unsupported data type ${e}`);return hn(s,0,1,e,t)}const jp=g({randomStandardNormal_:Bp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qp(s,e=0,t=1,n="float32",r){Ne(s);const a=Fe(s,n),i=new Vp(e,t,null,r);for(let o=0;o<a.values.length;o++)a.values[o]=i.nextValue();return a.toTensor()}const is=g({randomUniform_:qp});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kp(s,e,t,n){return is(s,e,t,"int32",n)}const Wp=g({randomUniformInt_:Kp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ye(s,e,t=1,n="float32"){if(t===0)throw new Error("Cannot have a step of zero");const r={start:s,stop:e,step:t,dtype:n};return b.runKernel(la,{},r)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hp(s){const t={input:h(s,"input","real")};return b.runKernel(ma,t)}const Qe=g({real_:Hp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Gp(s){const t={x:h(s,"x","reciprocal")};return b.runKernel(ha,t)}const Up=g({reciprocal_:Gp});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Xp(s,e){const n={x:h(s,"x","reverse")},r={dims:e};return b.runKernel(da,n,r)}const Ee=g({reverse_:Xp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yp(s){const e=h(s,"x","reverse");return y(e.rank===1,()=>`Error in reverse1D: x must be rank 1 but got rank ${e.rank}.`),Ee(e,0)}const Qp=g({reverse1d_:Yp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zp(s,e){const t=h(s,"x","reverse");return y(t.rank===2,()=>`Error in reverse2D: x must be rank 2 but got rank ${t.rank}.`),Ee(t,e)}const Jp=g({reverse2d_:Zp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mp(s,e){const t=h(s,"x","reverse");return y(t.rank===3,()=>`Error in reverse3D: x must be rank 3 but got rank ${t.rank}.`),Ee(t,e)}const el=g({reverse3d_:Mp});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tl(s,e){const t=h(s,"x","reverse");return y(t.rank===4,()=>`Error in reverse4D: x must be rank 4 but got rank ${t.rank}.`),Ee(t,e)}const sl=g({reverse4d_:tl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nl(s){const t={x:h(s,"x","round")};return b.runKernel(fa,t)}const dn=g({round_:nl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function rl(s){const t={x:h(s,"x","rsqrt","float32")};return b.runKernel(ya,t)}const al=g({rsqrt_:rl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ol(s){const t={x:h(s,"x","selu")};return b.runKernel(ga,t)}const il=g({selu_:ol});function ul(s,e,t,n,r,a=[1,1],i="NHWC"){const o=h(s,"x","separableConv2d"),u=h(e,"depthwiseFilter","separableConv2d"),p=h(t,"pointwiseFilter","separableConv2d");let l=o,m=!1;if(o.rank===3&&(m=!0,l=w(o,[1,o.shape[0],o.shape[1],o.shape[2]])),i==="NCHW")throw new Error("separableConv2d currently does not support dataFormat NCHW; only NHWC is supported");y(l.rank===4,()=>`Error in separableConv2d: input must be rank 4, but got rank ${l.rank}.`),y(u.rank===4,()=>`Error in separableConv2d: depthwise filter must be rank 4, but got rank ${u.rank}.`),y(p.rank===4,()=>`Error in separableConv2d: pointwise filter must be rank 4, but got rank ${u.rank}.`),y(p.shape[0]===1,()=>`Error in separableConv2d: the first dimension of pointwise filter  must be 1, but got ${p.shape[0]}.`),y(p.shape[1]===1,()=>`Error in separableConv2d: the second dimension of pointwise filter must be 1, but got ${p.shape[1]}.`);const d=u.shape[2],f=u.shape[3];y(p.shape[2]===d*f,()=>`Error in separableConv2d: the third dimension of pointwise filter must be ${d*f}, but got ${p.shape[2]}.`);const N=ss(l,u,n,r,i,a),T=_t(N,p,1,"valid",i);return m?w(T,[T.shape[1],T.shape[2],T.shape[3]]):T}const cl=g({separableConv2d_:ul});/**
 * @license
 * Copyright 2020 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function pl(s,e){const t=h(s,"x","setdiff1d"),n=h(e,"y","setdiff1d");y(t.dtype===n.dtype,()=>`x and y should have the same dtype, but got x (${t.dtype}) and y (${n.dtype}).`),y(t.rank===1,()=>`x should be 1D tensor, but got x (${t.shape}).`),y(n.rank===1,()=>`y should be 1D tensor, but got y (${n.shape}).`);const r=await t.data(),a=await n.data(),i=new Set(a);let o=0;for(let l=0;l<r.length;l++)i.has(r[l])||o++;const u=new ds([o],t.dtype),p=new ds([o],"int32");for(let l=0,m=0;l<r.length;l++)i.has(r[l])||(u.values[m]=r[l],p.values[m]=l,m++);return[u.toTensor(),p.toTensor()]}const ll=pl;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ml(s){const t={x:h(s,"x","sign")};return b.runKernel(ba,t)}const hl=g({sign_:ml});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dl(s){const t={x:h(s,"x","sin","float32")};return b.runKernel(Na,t)}const fl=g({sin_:dl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yl(s){const t={x:h(s,"x","sinh")};return b.runKernel(Ta,t)}const gl=g({sinh_:yl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bl(s,e,t){const n=h(s,"x","slice1d");return y(n.rank===1,()=>`slice1d expects a rank-1 tensor, but got a rank-${n.rank} tensor`),F(n,[e],[t])}const Nl=g({slice1d_:bl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tl(s,e,t){const n=h(s,"x","slice2d");return y(n.rank===2,()=>`slice2d expects a rank-2 tensor, but got a rank-${n.rank} tensor`),F(n,e,t)}const wl=g({slice2d_:Tl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $l(s,e,t){const n=h(s,"x","slice3d");return y(n.rank===3,()=>`slice3d expects a rank-3 tensor, but got a rank-${n.rank} tensor`),F(n,e,t)}const Sl=g({slice3d_:$l});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _l(s,e,t){const n=h(s,"x","slice4d");return y(n.rank===4,()=>`slice4d expects a rank-4 tensor, but got a rank-${n.rank} tensor`),F(n,e,t)}const vl=g({slice4d_:_l});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function kl(s,e=-1){const t=h(s,"logits","softmax","float32");if(e===-1&&(e=t.rank-1),e!==t.rank-1)throw Error(`Softmax along a non-last dimension is not yet supported. Logits was rank ${t.rank} and dim was ${e}`);const n={logits:t},r={dim:e};return b.runKernel(wa,n,r)}const El=g({softmax_:kl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xl(s){y(s.dtype==="complex64",()=>`The dtype for tf.spectral.fft() must be complex64 but got ${s.dtype}.`);const e={input:s};return b.runKernel($a,e)}const us=g({fft_:xl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ol(s){y(s.dtype==="complex64",()=>`The dtype for tf.spectral.ifft() must be complex64 but got ${s.dtype}.`);const e={input:s};return b.runKernel(Sa,e)}const ft=g({ifft_:Ol});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Il(s){const e=s.shape[s.shape.length-1],t=s.size/e;let n;if(e<=2){const r=w(s,[t,e]);n=ft(r)}else{const r=[t,2*(e-1)],a=w(Qe(s),[t,e]),i=w(Et(s),[t,e]),o=Ee(F(a,[0,1],[t,e-2]),1),u=I(Ee(F(i,[0,1],[t,e-2]),1),L(-1)),p=Y([a,o],1),l=Y([i,u],1),m=w(Le(p,l),[r[0],r[1]]);n=ft(m)}if(n=Qe(n),s.rank===3&&s.shape[0]!==0){const r=n,a=s.shape[0];n=w(n,[a,n.shape[0]/a,n.shape[1]]),r.dispose()}return n}const fn=g({irfft_:Il});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dl(s,e,t=0){const r={x:h(s,"x","split")},a={numOrSizeSplits:e,axis:t};return b.runKernel(_a,r,a)}const Ze=g({split_:Dl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Al(s,e){y(s.dtype==="float32",()=>`The dtype for rfft() must be real value but got ${s.dtype}`);let t=s.shape[s.shape.length-1];const n=s.size/t;let r;if(e!=null&&e<t){const N=s.shape.map(T=>0),$=s.shape.map(T=>T);$[s.shape.length-1]=e,r=F(s,N,$),t=e}else if(e!=null&&e>t){const N=s.shape.map($=>$);N[s.shape.length-1]=e-t,r=Y([s,$t(N)],s.shape.length-1),t=e}else r=s;const a=Xt(r),i=w(Le(r,a),[n,t]),o=us(i),u=Math.floor(t/2)+1,p=Qe(o),l=Et(o),m=Ze(p,[u,t-u],p.shape.length-1),d=Ze(l,[u,t-u],l.shape.length-1),f=r.shape.slice();return f[r.shape.length-1]=u,w(Le(m[0],d[0]),f)}const cs=g({rfft_:Al});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cl(s,e){let t=h(s,"a","squaredDifference"),n=h(e,"b","squaredDifference");[t,n]=Q(t,n),G(t.shape,n.shape);const r={a:t,b:n},a={};return b.runKernel(va,r,a)}const yn=g({squaredDifference_:Cl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zl(s,e){const t=h(s,"x","squeeze","string_or_numeric");return w(t,ka(t.shape,e).newShape)}const ps=g({squeeze_:zl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pl(s,e=0){const t=pt(s,"tensors","stack","string_or_numeric");y(t.length>=1,()=>"Pass at least one tensor to tf.stack"),t.length>0&&y(e<=t[0].rank,()=>"Axis must be <= rank of the tensor");const n=t,r={axis:e};return b.runKernel(Ea,n,r)}const de=g({stack_:Pl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vl(s,e,t,n,r=0,a=0,i=0,o=0,u=0){const l={x:h(s,"x","stridedSlice","string_or_numeric")},m={begin:e,end:t,strides:n,beginMask:r,endMask:a,ellipsisMask:i,newAxisMask:o,shrinkAxisMask:u};return b.runKernel(xa,l,m)}const Ll=g({stridedSlice_:Vl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fl(s){const t={x:h(s,"x","tan","float32")};return b.runKernel(Oa,t)}const Rl=g({tan_:Fl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function re(s,e){Re(s);const t=Be(s,e);if(t.length!==1)throw new Error("tensor1d() requires values to be a flat/TypedArray");return je(s,null,t,e)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ge(s,e,t){if(Re(s),e!=null&&e.length!==2)throw new Error("tensor2d() requires shape to have two numbers");const n=Be(s,t);if(n.length!==2&&n.length!==1)throw new Error("tensor2d() requires values to be number[][] or flat/TypedArray");if(n.length===1&&e==null)throw new Error("tensor2d() requires shape to be provided when `values` are a flat/TypedArray");return je(s,e,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function gn(s,e,t){if(Re(s),e!=null&&e.length!==3)throw new Error("tensor3d() requires shape to have three numbers");const n=Be(s,t);if(n.length!==3&&n.length!==1)throw new Error("tensor3d() requires values to be number[][][] or flat/TypedArray");if(n.length===1&&e==null)throw new Error("tensor3d() requires shape to be provided when `values` are a flat array");return je(s,e,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Bl(s,e,t){if(Re(s),e!=null&&e.length!==4)throw new Error("tensor4d() requires shape to have four numbers");const n=Be(s,t);if(n.length!==4&&n.length!==1)throw new Error("tensor4d() requires values to be number[][][][] or flat/TypedArray");if(n.length===1&&e==null)throw new Error("tensor4d() requires shape to be provided when `values` are a flat array");return je(s,e,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jl(s,e,t){if(Re(s),e!=null&&e.length!==5)throw new Error("tensor5d() requires shape to have five numbers");const n=Be(s,t);if(n.length!==5&&n.length!==1)throw new Error("tensor5d() requires values to be number[][][][][] or flat/TypedArray");if(n.length===1&&e==null)throw new Error("tensor5d() requires shape to be provided when `values` are a flat array");return je(s,e,n,t)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ql(s,e,t){if(Re(s),e!=null&&e.length!==6)throw new Error("tensor6d() requires shape to have six numbers");const n=Be(s,t);if(n.length!==6&&n.length!==1)throw new Error("tensor6d() requires values to be number[][][][][][] or flat/TypedArray");if(n.length===1&&e==null)throw new Error("tensor6d() requires shape to be provided when `values` are a flat array");return e=e||n,je(s,e,n,t)}/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kl(s,e,t){const n=h(s,"tensor","tensorScatterupdate"),r=h(e,"indices","tensorScatterupdate","int32"),a=h(t,"updates","tensorScatterupdate");if(Rs(a,r,n.shape),n.dtype!==a.dtype)throw new Error(`tensor and updates must have the same dtype, instead they are ${n.dtype} and ${a.dtype}.`);const i={tensor:n,indices:r,updates:a},o={};return b.runKernel(Ia,i,o)}const Wl=g({tensorScatterUpdate_:Kl});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hl(s,e=1,t=!0){const n=h(s,"x","topk");if(n.rank===0)throw new Error("topk() expects the input to be of rank 1 or higher");const r=n.shape[n.shape.length-1];if(e<0)throw new Error(`'k' passed to topk() must be >= 0 but got ${e}`);if(e>r)throw new Error(`'k' passed to topk() must be <= the last dimension (${r}) but got ${e}`);const a={x:n},i={k:e,sorted:t},[o,u]=b.runKernel(Da,a,i);return{values:o,indices:u}}const Gl=g({topk_:Hl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ul(s,e=0,t=1,n,r){if(Ne(s),n!=null&&n==="bool")throw new Error("Unsupported data type $ { dtype }");const a=new os(e,t,n,!0,r),i=Fe(s,n);for(let o=0;o<i.values.length;o++)i.values[o]=a.nextValue();return i.toTensor()}const Xl=g({truncatedNormal_:Ul});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yl(s,e=0){const t=h(s,"x","unique","string_or_numeric");y(t.rank>0,()=>"The input tensor must be at least 1D");const n={x:t},r={axis:e},[a,i]=b.runKernel(Aa,n,r);return{values:a,indices:i}}const Ql=g({unique_:Yl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zl(s,e,t){const n=h(s,"x","unsortedSegmentSum"),r=h(e,"segmentIds","unsortedSegmentSum","int32");y(Fs(t),()=>"numSegments must be of dtype int");const a={x:n,segmentIds:r},i={numSegments:t};return b.runKernel(Ca,a,i)}const Jl=g({unsortedSegmentSum_:Zl});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ml(s,e=0){const t=h(s,"x","unstack","string_or_numeric");y(e>=-t.shape.length&&e<t.shape.length,()=>`Axis = ${e} is not in [-${t.shape.length}, ${t.shape.length})`);const n={value:t},r={axis:e};return b.runKernel(za,n,r)}const Oe=g({unstack_:Ml});/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function em(s,e){return rs(s,e,"right")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function tm(s,e=!0,t,n){return b.makeVariable(s,e,t,n)}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function sm(s){const e=h(s,"condition","whereAsync","bool"),t=await e.data(),n=Pa(e.shape,t);return s!==e&&e.dispose(),n}const bn=sm;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function nm(s,e,t){const n=h(s,"tensor","boolMask"),r=h(e,"mask","boolMask","bool"),a=t??0,i=r.rank,o=n.shape;y(i>0,()=>"mask cannot be scalar"),ie(o.slice(a,a+i),r.shape,"mask's shape must match the first K dimensions of tensor's shape,");let u=1;for(let $=a;$<a+i;$++)u*=o[$];const p=o.slice(0,a).concat([u],o.slice(a+i)),l=w(n,p),m=w(r,[-1]),d=await bn(m),f=ps(d,[1]),N=sn(l,f,a);return s!==n&&n.dispose(),e!==r&&r.dispose(),f.dispose(),l.dispose(),m.dispose(),d.dispose(),N}const rm=nm;/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function am(s,e,t){const n=h(s,"x","transpose");if(e==null&&(e=n.shape.map((i,o)=>o).reverse()),y(n.rank===e.length,()=>`Error in transpose: rank of input ${n.rank} must match length of perm ${e}.`),e.forEach(i=>{y(i>=0&&i<n.rank,()=>`All entries in 'perm' must be between 0 and ${n.rank-1} but got ${e}`)}),n.rank<=1)return n.clone();const r={x:n},a={perm:e};return n.dtype==="complex64"?ae(()=>{let i=Qe(n),o=Et(n);return i=b.runKernel(xt,{x:i},a),o=b.runKernel(xt,{x:o},a),t&&(o=ue(o)),Le(i,o)}):b.runKernel(xt,r,a)}const Vt=g({transpose_:am});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function om(s,e,t,n,r=!0){const a=h(s,"v","movingAverage"),i=h(e,"x","movingAverage"),o=h(t,"decay","movingAverage");Va(a,i),y(Pe(a.shape,i.shape),()=>"Shape mismatch in v and x");const u=L(1),p=P(u,o);let l=I(P(i,a),p);if(r){y(n!=null,()=>"When using zeroDebias: true, step is required.");const m=h(n,"step","movingAverage");l=X(l,P(u,Qt(o,m)))}return W(a,l)}const im=g({movingAverage_:om});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function um(s,e,t){Ne(t);const n=h(s,"indices","scatterND","int32"),r=h(e,"updates","scatterND");Rs(r,n,t);const a={indices:n,updates:r},i={shape:t};return b.runKernel(La,a,i)}const cm=g({scatterND_:um});function pm(s,e,t,n){if(s.dtype!=="int32")throw new Error(`tf.sparseToDense() expects the indices to be int32 type, but the dtype was ${s.dtype}.`);if(s.rank>2)throw new Error(`sparseIndices should be a scalar, vector, or matrix, but got shape ${s.shape}.`);const r=s.rank>0?s.shape[0]:1,a=s.rank>1?s.shape[1]:1;if(t.length!==a)throw new Error(`outputShape has incorrect number of elements:, ${t.length}, should be: ${a}.`);const i=e.size;if(!(e.rank===0||e.rank===1&&i===r))throw new Error(`sparseValues has incorrect shape ${e.shape}, should be [] or [${r}]`);if(e.dtype!==n.dtype)throw new Error("sparseValues.dtype must match defaultValues.dtype")}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lm(s,e,t,n=0){Ne(t);const r=h(s,"sparseIndices","sparseToDense","int32"),a=h(e,"sparseValues","sparseToDense","string_or_numeric"),i=h(n,"defaultValue","sparseToDense",a.dtype);pm(r,a,t,i);const o={sparseIndices:r,sparseValues:a,defaultValue:i},u={outputShape:t};return b.runKernel(Fa,o,u)}const mm=g({sparseToDense_:lm});/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function hm(s,e){const t=h(e,"indices","gatherND","int32"),r={params:h(s,"x","gatherND","string_or_numeric"),indices:t};return b.runKernel(Ra,r)}const dm=g({gatherND_:hm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function fm(s,e){if(e==null)return s.shape.slice();if(Pe(s.shape,e))return e;if(s.shape.length===e.length){const t=[];for(let n=0;n<s.shape.length;n++)e[n]==null&&s.shape[n]!=null?t.push(s.shape[n]):t.push(e[n]);return t}return e}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ym(s,e,t,n){const r=h(s,"x","dropout");if(y(r.dtype==="float32",()=>`x has to be a floating point tensor since it's going to be scaled, but got a ${r.dtype} tensor instead.`),y(e>=0&&e<1,()=>`rate must be a float in the range [0, 1), but got ${e}.`),e===0)return s instanceof _e?r.clone():r;const a=fm(r,t),i=1-e,o=X(tn(W(is(a,0,1,"float32",n),i)),i);return I(r,o)}const gm=g({dropout_:ym});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Nn(s){return Math.floor(Math.pow(2,Math.ceil(Math.log(s)/Math.log(2))))}function ls(s,e,t){const n=1-s%2,r=new Float32Array(s);for(let a=0;a<s;++a){const i=2*Math.PI*a/(s+n-1);r[a]=e-t*Math.cos(i)}return re(r,"float32")}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function bm(s,e,t=1){const n=h(s,"predictions","inTopK"),r=h(e,"targets","inTopK");y(n.rank>1,()=>`inTopK() expects the predictions to be of rank 2 or higher, but got ${n.rank}`),y(n.rank-1===r.rank,()=>`predictions rank should be 1 larger than targets rank, but got predictions rank ${n.rank} and targets rank ${r.rank}`),ie(n.shape.slice(0,n.shape.length-1),r.shape,"predictions's shape should be align with the targets' shape, except the last dimension.");const a=n.shape[n.shape.length-1];y(t>0&&t<=a,()=>`'k' passed to inTopK() must be > 0 && <= the predictions last dimension (${a}), but got ${t}`);const i=await n.data(),o=await r.data(),[u,p]=[i.length/a,a],l=Ba("bool",u);for(let m=0;m<u;m++){const d=m*p,f=i.subarray(d,d+p),N=[];for(let $=0;$<f.length;$++)N.push({value:f[$],index:$});N.sort(($,T)=>T.value-$.value),l[m]=0;for(let $=0;$<t;$++)if(N[$].index===o[m]){l[m]=1;break}}return s!==n&&n.dispose(),e!==r&&r.dispose(),ve(l,r.shape,"bool")}const Nm=bm;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Tm(s,e,t,n,r,a="NHWC",i){let o=s;s.rank===3&&(o=w(s,[1,s.shape[0],s.shape[1],s.shape[2]]));let u=e;u.rank===3&&(u=w(e,[1,e.shape[0],e.shape[1],e.shape[2]])),y(o.rank===4,()=>`Error in conv2dDerFilter: input must be rank 4, but got shape ${o.shape}.`),y(u.rank===4,()=>`Error in conv2dDerFilter: dy must be rank 4, but got shape ${u.shape}.`),y(t.length===4,()=>`Error in conv2dDerFilter: filterShape must be length 4, but got ${t}.`);const p=a==="NHWC"?o.shape[3]:o.shape[1],l=a==="NHWC"?u.shape[3]:u.shape[1];y(p===t[2],()=>`Error in conv2dDerFilter: depth of input ${p}) must match input depth in filter (${t[2]}.`),y(l===t[3],()=>`Error in conv2dDerFilter: depth of dy (${l}) must match output depth for filter (${t[3]}).`),oe("conv2dDerFilter",r,i);const m={x:o,dy:u},d={strides:n,pad:r,dataFormat:a,dimRoundingMode:i,filterShape:t};return b.runKernel(ja,m,d)}const wm=g({conv2DBackpropFilter_:Tm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function $m({x:s,filter:e,strides:t,pad:n,dataFormat:r="NHWC",dilations:a=[1,1],dimRoundingMode:i,bias:o,activation:u="linear",preluActivationWeights:p,leakyreluAlpha:l}){if(u=u||"linear",Zt(b.state.gradientDepth,u)===!1){y(r==="NHWC",()=>`Error in fused conv2d: got dataFormat of ${r} but only NHWC is currently supported for the case of gradient depth is 0 and the activation is not linear.`);let v=_t(s,e,t,n,r,a,i);return o!=null&&(v=W(v,o)),Jt(v,u,p,l)}const m=h(s,"x","conv2d","float32"),d=h(e,"filter","conv2d","float32");let f=m,N=!1;m.rank===3&&(N=!0,f=w(m,[1,m.shape[0],m.shape[1],m.shape[2]])),y(f.rank===4,()=>`Error in fused conv2d: input must be rank 4, but got rank ${f.rank}.`),y(d.rank===4,()=>`Error in fused conv2d: filter must be rank 4, but got rank ${d.rank}.`),oe("fused conv2d",n,i);const $=r==="NHWC"?f.shape[3]:f.shape[1];y(d.shape[2]===$,()=>`Error in conv2d: depth of input (${$}) must match input depth for filter ${d.shape[2]}.`),y(be(t,a),()=>`Error in conv2D: Either strides or dilations must be 1. Got strides ${t} and dilations '${a}'`);const T=Bs(f.shape,d.shape,t,a,n,i);let S;o!=null&&(S=h(o,"bias","fused conv2d"),[S]=Q(S,m),r==="NHWC"?G(T.outShape,S.shape):(y(S.shape.length<=1,()=>`Error in fused conv2d: only supports scalar or 1-D Tensor bias for NCHW format but got the bias of rank-${S.shape.length}.`),y(S.shape.length===0||S.shape[0]===T.outChannels||S.shape[0]===1,()=>`Error in fused conv2d: bias shape (${S.shape}) is not compatible with the number of output channels (${T.outChannels})`)));let O;if(p!=null){const v=p.shape;if(y(v.length<=1||v.length===3,()=>`Error in fused conv2d: only supports scalar, 1-D Tensor or 3-D Tensor PReLU activation weights but got a tensor of rank-${v.length}.`),v.length===1)y(v[0]===1||v[0]===T.outChannels,()=>`Error in fused conv2d: PReLU activation weights (${v}) is not compatible with the number of output channels (${T.outChannels}).`);else if(v.length===3)try{G(v,T.outShape)}catch{const x=`Error in fused conv2d: PReLU activation weights (${v}) is not compatible with the output shape of the conv2d (${T.outShape}).`;throw Error(x)}O=h(p,"prelu weights","fused conv2d")}const C=(v,E)=>{y(r==="NHWC",()=>`Error in gradient of fused conv2D: got dataFormat of ${r} but only NHWC is currently supported.`);const[x,D,A,z]=E,j=Mt(v,A,u);y(js(a),()=>`Error in gradient of fused conv2D: dilation rates greater than 1 are not yet supported in gradients. Got dilations '${a}'`);const B=Zs(D.shape,j,x,t,n),q=wm(D,j,x.shape,t,n),Z=[B,q];if(z!=null){const te=es(z,j);Z.push(te)}return Z},_={x:f,filter:d,bias:S,preluActivationWeights:O},k={strides:t,pad:n,dataFormat:r,dilations:a,dimRoundingMode:i,activation:u,leakyreluAlpha:l};return o==null?he((E,x,D)=>{let A=b.runKernel(fs,_,k);return D([x,E,A]),N&&(A=w(A,[A.shape[1],A.shape[2],A.shape[3]])),{value:A,gradFunc:C}})(f,d):he((E,x,D,A)=>{let z=b.runKernel(fs,_,k);return A([x,E,z,D]),N&&(z=w(z,[z.shape[1],z.shape[2],z.shape[3]])),{value:z,gradFunc:C}})(f,d,S)}const Sm=g({fusedConv2d_:$m});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function _m(s,e,t,n,r,a=[1,1],i){let o=s;s.rank===3&&(o=w(s,[1,s.shape[0],s.shape[1],s.shape[2]]));let u=e;u.rank===3&&(u=w(e,[1,e.shape[0],e.shape[1],e.shape[2]]));const p={x:o,dy:u},l={strides:n,pad:r,dimRoundingMode:i,dilations:a,filterShape:t};return b.runKernel(qa,p,l)}const vm=g({depthwiseConv2dNativeBackpropFilter_:_m});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function km(s,e,t,n,r,a=[1,1],i){let o=e,u=!1;e.rank===3&&(u=!0,o=w(e,[1,e.shape[0],e.shape[1],e.shape[2]]));const p={dy:o,filter:t},l={strides:n,pad:r,dimRoundingMode:i,dilations:a,inputShape:s},m=b.runKernel(Ka,p,l);return u?w(m,[m.shape[1],m.shape[2],m.shape[3]]):m}const Em=g({depthwiseConv2dNativeBackpropInput_:km});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function xm({x:s,filter:e,strides:t,pad:n,dataFormat:r="NHWC",dilations:a=[1,1],dimRoundingMode:i,bias:o,activation:u="linear",preluActivationWeights:p,leakyreluAlpha:l}){if(Zt(b.state.gradientDepth,u)===!1){let k=ss(s,e,t,n,r,a,i);return o!=null&&(k=W(k,o)),Jt(k,u,p,l)}const m=h(s,"x","depthwiseConv2d","float32"),d=h(e,"filter","depthwiseConv2d","float32");let f=m,N=!1;m.rank===3&&(N=!0,f=w(m,[1,m.shape[0],m.shape[1],m.shape[2]])),y(f.rank===4,()=>`Error in fused depthwiseConv2d: input must be rank 4, but got rank ${f.rank}.`),y(d.rank===4,()=>`Error in fused depthwiseConv2d: filter must be rank 4, but got rank ${d.rank}.`),y(f.shape[3]===d.shape[2],()=>`Error in fused depthwiseConv2d: number of input channels (${f.shape[3]}) must match the inChannels dimension in filter ${d.shape[2]}.`),a==null&&(a=[1,1]),y(be(t,a),()=>`Error in fused depthwiseConv2d: Either strides or dilations must be 1. Got strides ${t} and dilations '${a}'`),oe("fused depthwiseConv2d",n,i);const $=Bs(f.shape,d.shape,t,a,n,i,!0);let T;o!=null&&(T=h(o,"bias","fused conv2d"),[T]=Q(T,m),G($.outShape,T.shape));let S;p!=null&&(S=h(p,"prelu weights","fused depthwiseConv2d"));const O=(k,v)=>{y(js(a),()=>`Error in gradient of fused depthwiseConv2d: dilation rates greater than 1 are not yet supported. Got dilations '${a}'`);const[E,x,D,A]=v,z=Mt(k,D,u),j=Em(x.shape,z,E,t,n,a,i),B=vm(x,z,E.shape,t,n,a,i);if(A!=null){const q=es(T,z);return[j,B,q]}return[j,B]},C={x:f,filter:d,bias:T,preluActivationWeights:S},_={strides:t,pad:n,dataFormat:r,dilations:a,dimRoundingMode:i,activation:u,leakyreluAlpha:l};return o==null?he((v,E,x)=>{let D=b.runKernel(ys,C,_);return x([E,v,D]),N&&(D=w(D,[D.shape[1],D.shape[2],D.shape[3]])),{value:D,gradFunc:O}})(f,d):he((v,E,x,D)=>{let A=b.runKernel(ys,C,_);return D([E,v,A,x]),N&&(A=w(A,[A.shape[1],A.shape[2],A.shape[3]])),{value:A,gradFunc:O}})(f,d,T)}const Om=g({fusedDepthwiseConv2d_:xm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Im({a:s,b:e,transposeA:t=!1,transposeB:n=!1,bias:r,activation:a="linear",preluActivationWeights:i,leakyreluAlpha:o=.2}){if(Zt(b.state.gradientDepth,a)===!1){let z=V(s,e,t,n);return r!=null&&(z=W(z,r)),Jt(z,a,i,o)}let u=h(s,"a","fused matMul"),p=h(e,"b","fused matMul");[u,p]=Q(u,p);const l=t?u.shape[u.rank-2]:u.shape[u.rank-1],m=n?p.shape[p.rank-1]:p.shape[p.rank-2],d=t?u.shape[u.rank-1]:u.shape[u.rank-2],f=n?p.shape[p.rank-2]:p.shape[p.rank-1],N=u.shape.slice(0,-2),$=p.shape.slice(0,-2),T=Se(N),S=Se($);y(l===m,()=>`Error in fused matMul: inner shapes (${l}) and (${m}) of Tensors with shapes ${u.shape} and ${p.shape} and transposeA=${t} and transposeB=${n} must match.`);const C=G(u.shape.slice(0,-2),p.shape.slice(0,-2)).concat([d,f]),_=t?w(u,[T,l,d]):w(u,[T,d,l]),k=n?w(p,[S,f,m]):w(p,[S,m,f]);let v;r!=null&&(v=h(r,"bias","fused matMul"),[v]=Q(v,u),G(C,v.shape));let E;i!=null&&(E=h(i,"prelu weights","fused matMul"));const x=(z,j)=>{const[B,q,Z,te]=j,ce=Mt(w(z,Z.shape),Z,a);let Ie,De;if(!t&&!n?(Ie=V(ce,q,!1,!0),De=V(B,ce,!0,!1)):!t&&n?(Ie=V(ce,q,!1,!1),De=V(ce,B,!0,!1)):t&&!n?(Ie=V(q,ce,!1,!0),De=V(B,ce,!1,!1)):(Ie=V(q,ce,!0,!0),De=V(ce,B,!0,!0)),r!=null){const In=es(te,ce);return[Ie,De,In]}else return[Ie,De]},D={a:_,b:k,bias:v,preluActivationWeights:E},A={transposeA:t,transposeB:n,activation:a,leakyreluAlpha:o};return r==null?he((j,B,q)=>{const Z=b.runKernel(gs,D,A);return q([j,B,Z]),{value:w(Z,C),gradFunc:x}})(_,k):he((j,B,q,Z)=>{const te=b.runKernel(gs,D,A);return Z([j,B,te,q]),{value:w(te,C),gradFunc:x}})(_,k,v)}const Dm=g({fusedMatMul_:Im});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Am=Object.freeze(Object.defineProperty({__proto__:null,conv2d:Sm,depthwiseConv2d:Om,matMul:Dm},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Cm(s){return ls(s,.54,.46)}const zm=g({hammingWindow_:Cm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Pm(s){return ls(s,.5,.5)}const Tn=g({hannWindow_:Pm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Vm(s,e,t,n=!1,r=0){let a=0;const i=[];for(;a+e<=s.size;)i.push(F(s,a,e)),a+=t;if(n)for(;a<s.size;){const o=a+e-s.size,u=Y([F(s,a,e-o),Nt([o],r)]);i.push(u),a+=t}return i.length===0?Ge([],[0,e]):w(Y(i),[i.length,e])}const wn=g({frame_:Vm});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Lm(s,e,t,n,r=Tn){n==null&&(n=Nn(e));const a=wn(s,e,t),i=I(a,r(e));return cs(i,n)}const Fm=g({stft_:Lm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Rm(s,e,t,n,r="bilinear",a=0){const i=h(s,"image","cropAndResize"),o=h(e,"boxes","cropAndResize","float32"),u=h(t,"boxInd","cropAndResize","int32"),p=o.shape[0];y(i.rank===4,()=>`Error in cropAndResize: image must be rank 4,but got rank ${i.rank}.`),y(o.rank===2&&o.shape[1]===4,()=>`Error in cropAndResize: boxes must be have size [${p},4] but had shape ${o.shape}.`),y(u.rank===1&&u.shape[0]===p,()=>`Error in cropAndResize: boxInd must be have size [${p}] but had shape ${o.shape}.`),y(n.length===2,()=>`Error in cropAndResize: cropSize must be of length 2, but got length ${n.length}.`),y(n[0]>=1&&n[1]>=1,()=>`cropSize must be atleast [1,1], but was ${n}`),y(r==="bilinear"||r==="nearest",()=>`method must be bilinear or nearest, but was ${r}`);const l={image:i,boxes:o,boxInd:u},m={method:r,extrapolationValue:a,cropSize:n};return b.runKernel(Wa,l,m)}const Bm=g({cropAndResize_:Rm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jm(s){const e=h(s,"image","flipLeftRight","float32");y(e.rank===4,()=>`Error in flipLeftRight: image must be rank 4,but got rank ${e.rank}.`);const t={image:e};return b.runKernel(Ha,t,{})}const qm=g({flipLeftRight_:jm});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Km(s){const e=h(s,"image","grayscaleToRGB"),t=e.rank-1,n=e.shape[t];y(e.rank>=2,()=>`Error in grayscaleToRGB: images must be at least rank 2, but got rank ${e.rank}.`),y(n===1,()=>`Error in grayscaleToRGB: last dimension of a grayscale image should be size 1, but got size ${n}.`);const r=new Array(e.rank);return r.fill(1,0,t),r[t]=3,He(e,r)}const Wm=g({grayscaleToRGB_:Km});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hm(s){const e=h(s,"image","RGBToGrayscale"),t=e.rank-1,n=e.shape[t];y(e.rank>=2,()=>`Error in RGBToGrayscale: images must be at least rank 2, but got rank ${e.rank}.`),y(n===3,()=>`Error in RGBToGrayscale: last dimension of an RGB image should be size 3, but got size ${n}.`);const r=e.dtype,a=K(e,"float32"),i=re([.2989,.587,.114]);let o;switch(e.rank){case 2:o=Ae("ij,j->i",a,i);break;case 3:o=Ae("ijk,k->ij",a,i);break;case 4:o=Ae("ijkl,l->ijk",a,i);break;case 5:o=Ae("ijklm,m->ijkl",a,i);break;case 6:o=Ae("ijklmn,n->ijklm",a,i);break;default:throw new Error("Not a valid tensor rank.")}return o=ye(o,-1),K(o,r)}const Gm=g({rgbToGrayscale_:Hm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Um(s,e,t=0,n=.5){const r=h(s,"image","rotateWithOffset","float32");y(r.rank===4,()=>`Error in rotateWithOffset: image must be rank 4,but got rank ${r.rank}.`);const a={image:r},i={radians:e,fillValue:t,center:n};return b.runKernel(Ga,a,i)}const Xm=g({rotateWithOffset_:Um});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function qe(s,e,t,n,r,a){n==null&&(n=.5),r==null&&(r=Number.NEGATIVE_INFINITY),a==null&&(a=0);const i=s.shape[0];return t=Math.min(t,i),y(0<=n&&n<=1,()=>`iouThreshold must be in [0, 1], but was '${n}'`),y(s.rank===2,()=>`boxes must be a 2D tensor, but was of rank '${s.rank}'`),y(s.shape[1]===4,()=>`boxes must have 4 columns, but 2nd dimension was ${s.shape[1]}`),y(e.rank===1,()=>"scores must be a 1D tensor"),y(e.shape[0]===i,()=>`scores has incompatible shape with boxes. Expected ${i}, but was ${e.shape[0]}`),y(0<=a&&a<=1,()=>`softNmsSigma must be in [0, 1], but was '${a}'`),{maxOutputSize:t,iouThreshold:n,scoreThreshold:r,softNmsSigma:a}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ym(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY){const a=h(s,"boxes","nonMaxSuppression","float32"),i=h(e,"scores","nonMaxSuppression","float32"),o=qe(a,i,t,n,r);t=o.maxOutputSize,n=o.iouThreshold,r=o.scoreThreshold;const u={maxOutputSize:t,iouThreshold:n,scoreThreshold:r};return b.runKernel(Ua,{boxes:a,scores:i},u)}const Qm=g({nonMaxSuppression_:Ym});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function Zm(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY){const a=h(s,"boxes","nonMaxSuppressionAsync"),i=h(e,"scores","nonMaxSuppressionAsync"),o=qe(a,i,t,n,r);t=o.maxOutputSize,n=o.iouThreshold,r=o.scoreThreshold;const u=await Promise.all([a.data(),i.data()]),p=u[0],l=u[1],{selectedIndices:m}=Xa(p,l,t,n,r);return a!==s&&a.dispose(),i!==e&&i.dispose(),re(m,"int32")}const Jm=Zm;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mm(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,a=0){const i=h(s,"boxes","nonMaxSuppression"),o=h(e,"scores","nonMaxSuppression"),u=qe(i,o,t,n,r,a);t=u.maxOutputSize,n=u.iouThreshold,r=u.scoreThreshold,a=u.softNmsSigma;const p={boxes:i,scores:o},l={maxOutputSize:t,iouThreshold:n,scoreThreshold:r,softNmsSigma:a},m=b.runKernel(Ya,p,l);return{selectedIndices:m[0],selectedScores:m[1]}}const eh=g({nonMaxSuppressionWithScore_:Mm});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function th(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,a=0){const i=h(s,"boxes","nonMaxSuppressionAsync"),o=h(e,"scores","nonMaxSuppressionAsync"),u=qe(i,o,t,n,r,a);t=u.maxOutputSize,n=u.iouThreshold,r=u.scoreThreshold,a=u.softNmsSigma;const p=await Promise.all([i.data(),o.data()]),l=p[0],m=p[1],{selectedIndices:d,selectedScores:f}=Qa(l,m,t,n,r,a);return i!==s&&i.dispose(),o!==e&&o.dispose(),{selectedIndices:re(d,"int32"),selectedScores:re(f)}}const sh=th;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function nh(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,a=!1){const i=h(s,"boxes","nonMaxSuppression"),o=h(e,"scores","nonMaxSuppression"),u=qe(i,o,t,n,r,null),p=u.maxOutputSize,l=u.iouThreshold,m=u.scoreThreshold,d={boxes:i,scores:o},f={maxOutputSize:p,iouThreshold:l,scoreThreshold:m,padToMaxOutputSize:a},N=b.runKernel(Za,d,f);return{selectedIndices:N[0],validOutputs:N[1]}}const rh=g({nonMaxSuppressionPadded_:nh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */async function ah(s,e,t,n=.5,r=Number.NEGATIVE_INFINITY,a=!1){const i=h(s,"boxes","nonMaxSuppressionAsync"),o=h(e,"scores","nonMaxSuppressionAsync"),u=qe(i,o,t,n,r,null),p=u.maxOutputSize,l=u.iouThreshold,m=u.scoreThreshold,[d,f]=await Promise.all([i.data(),o.data()]),{selectedIndices:N,validOutputs:$}=Ja(d,f,p,l,m,a);return i!==s&&i.dispose(),o!==e&&o.dispose(),{selectedIndices:re(N,"int32"),validOutputs:L($,"int32")}}const oh=ah;/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ih(s,e,t=!1,n=!1){const r=h(s,"images","resizeBilinear");y(r.rank===3||r.rank===4,()=>`Error in resizeBilinear: x must be rank 3 or 4, but got rank ${r.rank}.`),y(e.length===2,()=>`Error in resizeBilinear: new shape must 2D, but got shape ${e}.`),y(n===!1||t===!1,()=>"Error in resizeBilinear: If halfPixelCenters is true, alignCorners must be false.");let a=r,i=!1;r.rank===3&&(i=!0,a=w(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const o={images:a},u={alignCorners:t,halfPixelCenters:n,size:e},p=b.runKernel(Ma,o,u);return i?w(p,[p.shape[1],p.shape[2],p.shape[3]]):p}const uh=g({resizeBilinear_:ih});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ch(s,e,t=!1,n=!1){const r=h(s,"images","resizeNearestNeighbor");y(r.rank===3||r.rank===4,()=>`Error in resizeNearestNeighbor: x must be rank 3 or 4, but got rank ${r.rank}.`),y(e.length===2,()=>`Error in resizeNearestNeighbor: new shape must 2D, but got shape ${e}.`),y(r.dtype==="float32"||r.dtype==="int32",()=>"`images` must have `int32` or `float32` as dtype"),y(n===!1||t===!1,()=>"Error in resizeNearestNeighbor: If halfPixelCenters is true, alignCorners must be false.");let a=r,i=!1;r.rank===3&&(i=!0,a=w(r,[1,r.shape[0],r.shape[1],r.shape[2]]));const o={images:a},u={alignCorners:t,halfPixelCenters:n,size:e},p=b.runKernel(eo,o,u);return i?w(p,[p.shape[1],p.shape[2],p.shape[3]]):p}const ph=g({resizeNearestNeighbor_:ch});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function lh(s,e="binary",t=!1,n=.5){const r=h(s,"image","threshold"),a=.2989,i=.587,o=.114,u=r.shape[0]*r.shape[1];let p=I(re([n]),255),l,m,d,f;if(y(r.rank===3,()=>`Error in threshold: image must be rank 3,but got rank ${r.rank}.`),y(r.shape[2]===3||r.shape[2]===1,()=>`Error in threshold: image color channel must be equal to 3 or 1but got ${r.shape[2]}.`),y(r.dtype==="int32"||r.dtype==="float32",()=>`Error in dtype: image dtype must be int32 or float32,but got dtype ${r.dtype}.`),y(e==="otsu"||e==="binary",()=>`Method must be binary or otsu, but was ${e}`),r.shape[2]===3){[l,m,d]=Ze(r,[1,1,1],-1);const T=I(l,a),S=I(m,i),O=I(d,o);f=W(W(T,S),O)}else f=s;if(e==="otsu"){const T=Qs(K(dn(f),"int32"),ve([]),256);p=mh(T,u)}const N=t?ns(f,p):kt(f,p);return K(I(N,255),"int32")}function mh(s,e){let t=re([-1]),n=re([0]),r=re([0]),a,i,o,u,p,l;for(let m=0;m<s.size-1;m++){a=F(s,0,m+1),i=F(s,m+1),p=X(R(a),e),l=X(R(i),e);const d=R(I(a,Ye(0,a.size)));o=X(d,R(a));const f=Nt(i.shape,a.size),N=W(Ye(0,i.size),f),$=I(i,N);u=X(R($),R(i));const T=P(o,u),S=P(o,u),O=I(p,l);r=I(I(O,T),S);const C=kt(r,n);n=ge(C,r,n),t=ge(C,re([m]),t)}return t}const hh=g({threshold_:lh});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function dh(s,e,t="nearest",n="constant",r=0,a){const i=h(s,"image","transform","float32"),o=h(e,"transforms","transform","float32");y(i.rank===4,()=>`Error in transform: image must be rank 4,but got rank ${i.rank}.`),y(o.rank===2&&(o.shape[0]===i.shape[0]||o.shape[0]===1)&&o.shape[1]===8,()=>"Error in transform: Input transform should be batch x 8 or 1 x 8"),y(a==null||a.length===2,()=>`Error in transform: outputShape must be [height, width] or null, but got ${a}.`);const u={image:i,transforms:o},p={interpolation:t,fillMode:n,fillValue:r,outputShape:a};return b.runKernel(to,u,p)}const fh=g({transform_:dh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function yh(s,e,t){const n=h(s,"a","bandPart");y(n.rank>=2,()=>`bandPart(): Rank must be at least 2, got ${n.rank}.`);const r=n.shape,[a,i]=n.shape.slice(-2);let o,u;typeof e=="number"?(y(e%1===0,()=>`bandPart(): numLower must be an integer, got ${e}.`),y(e<=a,()=>`bandPart(): numLower (${e}) must not be greater than the number of rows (${a}).`),o=h(e<0?a:e,"numLower","bandPart")):(y(e.dtype==="int32",()=>"bandPart(): numLower's dtype must be an int32."),o=ge(Pt(e,0),a,dt(e,a))),typeof t=="number"?(y(t%1===0,()=>`bandPart(): numUpper must be an integer, got ${t}.`),y(t<=i,()=>`bandPart(): numUpper (${t}) must not be greater than the number of columns (${i}).`),u=h(t<0?i:t,"numUpper","bandPart")):(y(t.dtype==="int32",()=>"bandPart(): numUpper's dtype must be an int32."),u=ge(Pt(t,0),i,dt(t,i)));const p=w(Ye(0,a,1,"int32"),[-1,1]),l=Ye(0,i,1,"int32"),m=P(p,l),d=mt(ns(m,o),nn(m,ue(u))),f=$t([a,i],n.dtype);return w(de(Oe(w(n,[-1,a,i])).map(N=>ge(d,N,f))),r)}const gh=g({bandPart_:yh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function bh(s){let e;if(Array.isArray(s)){e=!1,y(s!=null&&s.length>0,()=>"Gram-Schmidt process: input must not be null, undefined, or empty");const r=s[0].shape[0];for(let a=1;a<s.length;++a)y(s[a].shape[0]===r,()=>`Gram-Schmidt: Non-unique lengths found in the input vectors: (${s[a].shape[0]} vs. ${r})`)}else e=!0,s=Ze(s,s.shape[0],0).map(r=>ps(r,[0]));y(s.length<=s[0].shape[0],()=>`Gram-Schmidt: Number of vectors (${s.length}) exceeds number of dimensions (${s[0].shape[0]}).`);const t=[],n=s;for(let r=0;r<s.length;++r)t.push(b.tidy(()=>{let a=n[r];if(r>0)for(let i=0;i<r;++i){const o=I(R(I(t[i],a)),t[i]);a=P(a,o)}return X(a,vt(a,"euclidean"))}));return e?de(t,0):t}const Nh=g({gramSchmidt_:bh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Th(s,e=!1){if(y(s.rank>=2,()=>`qr() requires input tensor to have a rank >= 2, but got rank ${s.rank}`),s.rank===2)return Es(s,e);{const t=s.shape.slice(0,s.shape.length-2).reduce((u,p)=>u*p),n=Oe(w(s,[t,s.shape[s.shape.length-2],s.shape[s.shape.length-1]]),0),r=[],a=[];n.forEach(u=>{const[p,l]=Es(u,e);r.push(p),a.push(l)});const i=w(de(r,0),s.shape),o=w(de(a,0),s.shape);return[i,o]}}function Es(s,e=!1){return b.tidy(()=>{y(s.shape.length===2,()=>`qr2d() requires a 2D Tensor, but got a ${s.shape.length}D Tensor.`);const t=s.shape[0],n=s.shape[1];let r=en(t),a=Ce(s);const i=Ge([[1]],[1,1]);let o=Ce(i);const u=t>=n?n:t;for(let p=0;p<u;++p){const l=a,m=o,d=r;[o,a,r]=b.tidy(()=>{const f=F(a,[p,p],[t-p,1]),N=vt(f),$=F(a,[p,p],[1,1]),T=ge(kt($,0),Ge([[-1]]),Ge([[1]])),S=P($,I(T,N)),O=X(f,S);O.shape[0]===1?o=Ce(i):o=Y([i,F(O,[1,0],[O.shape[0]-1,O.shape[1]])],0);const C=ue(X(V(T,S),N)),_=F(a,[p,0],[t-p,n]),k=I(C,o),v=Vt(o);if(p===0)a=P(_,V(k,V(v,_)));else{const D=P(_,V(k,V(v,_)));a=Y([F(a,[0,0],[p,n]),D],0)}const E=Vt(k),x=F(r,[0,p],[t,r.shape[1]-p]);if(p===0)r=P(x,V(V(x,o),E));else{const D=P(x,V(V(x,o),E));r=Y([F(r,[0,0],[t,p]),D],1)}return[o,a,r]}),qs([l,m,d])}return!e&&t>n&&(r=F(r,[0,0],[t,n]),a=F(a,[0,0],[n,n])),[r,a]})}const wh=g({qr_:Th});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var J;(function(s){s[s.NONE=0]="NONE",s[s.MEAN=1]="MEAN",s[s.SUM=2]="SUM",s[s.SUM_BY_NONZERO_WEIGHTS=3]="SUM_BY_NONZERO_WEIGHTS"})(J||(J={}));function $h(s,e,t=J.SUM_BY_NONZERO_WEIGHTS){const n=h(s,"losses","computeWeightedLoss");let r=null;e!=null&&(r=h(e,"weights","computeWeightedLoss"));const a=r==null?n:I(n,r);if(t===J.NONE)return a;if(t===J.SUM)return R(a);if(t===J.MEAN){if(r==null)return ht(a);{const i=n.size/r.size,o=X(R(a),R(r));return i>1?X(o,L(i)):o}}if(t===J.SUM_BY_NONZERO_WEIGHTS){if(r==null)return X(R(a),L(n.size));{const i=I(r,$e(n.shape)),o=K(R(ln(i,L(0))),"float32");return X(R(a),o)}}throw Error(`Unknown reduction: ${t}`)}const fe=g({computeWeightedLoss_:$h});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Sh(s,e,t,n=J.SUM_BY_NONZERO_WEIGHTS){const r=h(s,"labels","absoluteDifference"),a=h(e,"predictions","absoluteDifference");let i=null;t!=null&&(i=h(t,"weights","absoluteDifference")),ie(r.shape,a.shape,"Error in absoluteDifference: ");const o=se(P(r,a));return fe(o,i,n)}const _h=g({absoluteDifference_:Sh});function vh(s,e,t,n,r=J.SUM_BY_NONZERO_WEIGHTS){const a=h(s,"labels","cosineDistance"),i=h(e,"predictions","cosineDistance");let o=null;n!=null&&(o=h(n,"weights","cosineDistance")),ie(a.shape,i.shape,"Error in cosineDistance: ");const u=L(1),p=P(u,R(I(a,i),t,!0));return fe(p,o,r)}const kh=g({cosineDistance_:vh});function Eh(s,e,t,n=J.SUM_BY_NONZERO_WEIGHTS){let r=h(s,"labels","hingeLoss");const a=h(e,"predictions","hingeLoss");let i=null;t!=null&&(i=h(t,"weights","hingeLoss")),ie(r.shape,a.shape,"Error in hingeLoss: ");const o=L(1);r=P(I(L(2),r),o);const u=ts(P(o,I(r,a)));return fe(u,i,n)}const xh=g({hingeLoss_:Eh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oh(s,e,t,n=1,r=J.SUM_BY_NONZERO_WEIGHTS){const a=h(s,"labels","huberLoss"),i=h(e,"predictions","huberLoss");let o=null;t!=null&&(o=h(t,"weights","huberLoss")),ie(a.shape,i.shape,"Error in huberLoss: ");const u=L(n),p=se(P(i,a)),l=dt(p,u),m=P(p,l),d=W(I(L(.5),wt(l)),I(u,m));return fe(d,o,r)}const Ih=g({huberLoss_:Oh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dh(s,e,t,n=1e-7,r=J.SUM_BY_NONZERO_WEIGHTS){const a=h(s,"labels","logLoss"),i=h(e,"predictions","logLoss");let o=null;t!=null&&(o=h(t,"weights","logLoss")),ie(a.shape,i.shape,"Error in logLoss: ");const u=L(1),p=L(n),l=ue(I(a,Xe(W(i,p)))),m=I(P(u,a),Xe(W(P(u,i),p))),d=P(l,m);return fe(d,o,r)}const Ah=g({logLoss_:Dh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ch(s,e,t,n=J.SUM_BY_NONZERO_WEIGHTS){const r=h(s,"labels","meanSquaredError"),a=h(e,"predictions","meanSquaredError");let i=null;t!=null&&(i=h(t,"weights","meanSquaredError")),ie(r.shape,a.shape,"Error in meanSquaredError: ");const o=yn(r,a);return fe(o,i,n)}const zh=g({meanSquaredError_:Ch});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ph(s,e){const t=h(s,"labels","sigmoidCrossEntropyWithLogits"),n=h(e,"logits","sigmoidCrossEntropyWithLogits");ie(t.shape,n.shape,"Error in sigmoidCrossEntropyWithLogits: ");const r=ts(n),a=I(n,t),i=rn(ke(ue(se(n))));return W(P(r,a),i)}function Vh(s,e,t,n=0,r=J.SUM_BY_NONZERO_WEIGHTS){let a=h(s,"multiClassLabels","sigmoidCrossEntropy");const i=h(e,"logits","sigmoidCrossEntropy");let o=null;if(t!=null&&(o=h(t,"weights","sigmoidCrossEntropy")),ie(a.shape,i.shape,"Error in sigmoidCrossEntropy: "),n>0){const p=L(n),l=L(1),m=L(.5);a=W(I(a,P(l,p)),I(m,p))}const u=Ph(a,i);return fe(u,o,r)}const Lh=g({sigmoidCrossEntropy_:Vh});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Fh(s,e,t=-1){if(t===-1&&(t=e.rank-1),t!==e.rank-1)throw Error(`Softmax cross entropy along a non-last dimension is not yet supported. Labels / logits was rank ${e.rank} and dim was ${t}`);return he((r,a,i)=>{const u=on(a,[t],!0),p=P(K(a,"float32"),u);i([r,p]);const l=ue(I(p,r));return{value:R(l,[t]),gradFunc:(f,N)=>{const[$,T]=N,S=Tt(f.shape,[t]);return[I(w(f,S),P(K($,"float32"),ke(T))),I(w(f,S),P(ke(T),K($,"float32")))]}}})(s,e)}function Rh(s,e,t,n=0,r=J.SUM_BY_NONZERO_WEIGHTS){let a=h(s,"onehotLabels","softmaxCrossEntropy");const i=h(e,"logits","softmaxCrossEntropy");let o=null;if(t!=null&&(o=h(t,"weights","softmaxCrossEntropy")),ie(a.shape,i.shape,"Error in softmaxCrossEntropy: "),n>0){const p=L(n),l=L(1),m=L(a.shape[1]);a=W(I(a,P(l,p)),X(p,m))}const u=Fh(a,i);return fe(u,o,r)}const Bh=g({softmaxCrossEntropy_:Rh});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function jh(s,e,t,n){const r=h(s,"indices","sparseFillEmptyRows","int32"),a=h(e,"values","sparseFillEmptyRows"),i=h(t,"denseShape","sparseFillEmptyRows","int32"),o=h(n,"defaultValue","sparseFillEmptyRows",a.dtype);if(r.rank!==2)throw new Error(`Indices should be Tensor2D but received shape
        ${r.shape}`);if(a.rank!==1)throw new Error(`Values should be Tensor1D but received shape ${a.shape}`);if(i.rank!==1)throw new Error(`Dense shape should be Tensor1D but received shape ${i.shape}`);if(o.rank!==0)throw new Error(`Default value should be a scalar but received shape ${o.shape}`);const u={indices:r,values:a,denseShape:i,defaultValue:o},p=b.runKernel(so,u);return{outputIndices:p[0],outputValues:p[1],emptyRowIndicator:p[2],reverseIndexMap:p[3]}}const qh=g({sparseFillEmptyRows_:jh});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Kh(s,e,t){const n=h(s,"inputIndices","sparseReshape","int32"),r=h(e,"inputShape","sparseReshape","int32"),a=h(t,"newShape","sparseReshape","int32");if(n.rank!==2)throw new Error(`Input indices should be Tensor2D but received shape
        ${n.shape}`);if(r.rank!==1)throw new Error(`Input shape should be Tensor1D but received shape ${r.shape}`);if(a.rank!==1)throw new Error(`New shape should be Tensor1D but received shape ${a.shape}`);const i={inputIndices:n,inputShape:r,newShape:a},o=b.runKernel(no,i);return{outputIndices:o[0],outputShape:o[1]}}const Wh=g({sparseReshape_:Kh});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Hh(s,e,t){const n=h(s,"data","sparseSegmentMean"),r=h(e,"indices","sparseSegmentMean","int32"),a=h(t,"segmentIds","sparseSegmentMean","int32");if(n.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
          ${r.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
          ${a.shape}`);const i={data:n,indices:r,segmentIds:a};return b.runKernel(ro,i)}const Gh=g({sparseSegmentMean_:Hh});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Uh(s,e,t){const n=h(s,"data","sparseSegmentSum"),r=h(e,"indices","sparseSegmentSum","int32"),a=h(t,"segmentIds","sparseSegmentSum","int32");if(n.rank<1)throw new Error("Data should be at least 1 dimensional but received scalar");if(r.rank!==1)throw new Error(`Indices should be Tensor1D but received shape
         ${r.shape}`);if(a.rank!==1)throw new Error(`Segment ids should be Tensor1D but received shape
         ${a.shape}`);const i={data:n,indices:r,segmentIds:a};return b.runKernel(ao,i)}const Xh=g({sparseSegmentSum_:Uh});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Yh(s,e,t,n,r,a,i,o){const u=h(s,"data","stringNGrams","string");if(u.dtype!=="string")throw new Error("Data must be of datatype string");if(u.shape.length!==1)throw new Error(`Data must be a vector, saw: ${u.shape}`);const p=h(e,"dataSplits","stringNGrams");if(p.dtype!=="int32")throw new Error("Data splits must be of datatype int32");const l={separator:t,nGramWidths:n,leftPad:r,rightPad:a,padWidth:i,preserveShortSequences:o},m={data:u,dataSplits:p},d=b.runKernel(oo,m,l);return{nGrams:d[0],nGramsSplits:d[1]}}const Qh=g({stringNGrams_:Yh});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zh(s,e,t=!0){const n=h(s,"input","stringSplit","string"),r=h(e,"delimiter","stringSplit","string");if(n.rank!==1)throw new Error(`Input should be Tensor1D but received shape ${n.shape}`);if(r.rank!==0)throw new Error(`Delimiter should be a scalar but received shape ${r.shape}`);const a={skipEmpty:t},i={input:n,delimiter:r},o=b.runKernel(io,i,a);return{indices:o[0],values:o[1],shape:o[2]}}const Jh=g({stringSplit_:Zh});/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Mh(s,e){const t=h(s,"input","stringToHashBucketFast","string"),n={numBuckets:e};if(e<=0)throw new Error("Number of buckets must be at least 1");const r={input:t};return b.runKernel(uo,r,n)}const ed=g({stringToHashBucketFast_:Mh});/**
 * @license
 * Copyright 2023 Google LLC.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function td(s,e,t,n=!0){const r=h(s,"input","staticRegexReplace","string"),a={pattern:e,rewrite:t,replaceGlobal:n};return b.runKernel(co,{x:r},a)}const sd=g({staticRegexReplace_:td});/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const nd={fft:us,ifft:ft,rfft:cs,irfft:fn},rd={hammingWindow:zm,hannWindow:Tn,frame:wn,stft:Fm},ad={flipLeftRight:qm,grayscaleToRGB:Wm,resizeNearestNeighbor:ph,resizeBilinear:uh,rgbToGrayscale:Gm,rotateWithOffset:Xm,cropAndResize:Bm,nonMaxSuppression:Qm,nonMaxSuppressionAsync:Jm,nonMaxSuppressionWithScore:eh,nonMaxSuppressionWithScoreAsync:sh,nonMaxSuppressionPadded:rh,nonMaxSuppressionPaddedAsync:oh,threshold:hh,transform:fh},od={bandPart:gh,gramSchmidt:Nh,qr:wh},id={absoluteDifference:_h,computeWeightedLoss:fe,cosineDistance:kh,hingeLoss:xh,huberLoss:Ih,logLoss:Ah,meanSquaredError:zh,sigmoidCrossEntropy:Lh,softmaxCrossEntropy:Bh},ud={sparseFillEmptyRows:qh,sparseReshape:Wh,sparseSegmentMean:Gh,sparseSegmentSum:Xh},cd={stringNGrams:Qh,stringSplit:Jh,stringToHashBucketFast:ed,staticRegexReplace:sd};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class It{constructor(e){this.modelArtifacts=e}load(){return this.modelArtifacts}}class $n{constructor(e){this.saveHandler=e}save(e){return this.saveHandler(e)}}class pd{constructor(e){e.load&&(this.load=()=>Promise.resolve(e.load())),e.save&&(this.save=t=>Promise.resolve(e.save(t)))}}function ld(s,e,t,n){const r=arguments;return new pd(yt(...r))}function yt(s,e,t,n){return arguments.length===1?s.modelTopology!=null||s.weightSpecs!=null?new It(s):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new It({modelTopology:s})):(console.warn("Please call tf.io.fromMemory() with only one argument. The argument should be of type ModelArtifacts. The multi-argument signature of tf.io.fromMemory() has been deprecated and will be removed in a future release."),new It({modelTopology:s,weightSpecs:e,weightData:t,trainingConfig:n}))}function md(s){return new $n(s)}function hd(s){return new $n(s)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sn=Object.freeze(Object.defineProperty({__proto__:null,CompositeArrayBuffer:po,browserFiles:lo,browserHTTPRequest:mo,concatenateArrayBuffers:ho,copyModel:fo,decodeWeights:yo,decodeWeightsStream:Ks,encodeWeights:go,fromMemory:ld,fromMemorySync:yt,getLoadHandlers:bo,getModelArtifactsForJSON:No,getModelArtifactsForJSONSync:Ws,getModelArtifactsInfoForJSON:To,getSaveHandlers:wo,getWeightSpecs:Hs,http:$o,isHTTPScheme:So,listModels:_o,loadWeights:vo,moveModel:ko,registerLoadRouter:Eo,registerSaveRouter:xo,removeModel:Oo,weightsLoaderFactory:Io,withSaveHandler:md,withSaveHandlerSync:hd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */let Te,xs=!1;function _n(s,e=3){if(e>4)throw new Error("Cannot construct Tensor with more than 4 channels from pixels.");if(s==null)throw new Error("pixels passed to tf.browser.fromPixels() can not be null");let t=!1,n=!1,r=!1,a=!1,i=!1,o=!1;if(s.data instanceof Uint8Array)t=!0;else if(typeof ImageData<"u"&&s instanceof ImageData)n=!0;else if(typeof HTMLVideoElement<"u"&&s instanceof HTMLVideoElement)r=!0;else if(typeof HTMLImageElement<"u"&&s instanceof HTMLImageElement)a=!0;else if(s.getContext!=null)i=!0;else if(typeof ImageBitmap<"u"&&s instanceof ImageBitmap)o=!0;else throw new Error(`pixels passed to tf.browser.fromPixels() must be either an HTMLVideoElement, HTMLImageElement, HTMLCanvasElement, ImageData in browser, or OffscreenCanvas, ImageData in webworker or {data: Uint32Array, width: number, height: number}, but was ${s.constructor.name}`);if(Us(bs,b.backendName)!=null){const N={pixels:s},$={numChannels:e};return b.runKernel(bs,N,$)}const[p,l]=r?[s.videoWidth,s.videoHeight]:[s.width,s.height];let m;if(i)m=s.getContext("2d").getImageData(0,0,p,l).data;else if(n||t)m=s.data;else if(a||r||o){if(Te==null)if(typeof document>"u")if(typeof OffscreenCanvas<"u"&&typeof OffscreenCanvasRenderingContext2D<"u")Te=new OffscreenCanvas(1,1).getContext("2d");else throw new Error("Cannot parse input in current context. Reason: OffscreenCanvas Context2D rendering is not supported.");else Te=document.createElement("canvas").getContext("2d",{willReadFrequently:!0});Te.canvas.width=p,Te.canvas.height=l,Te.drawImage(s,0,0,p,l),m=Te.getImageData(0,0,p,l).data}let d;if(e===4)d=new Int32Array(m);else{const N=p*l;d=new Int32Array(N*e);for(let $=0;$<N;$++)for(let T=0;T<e;++T)d[$*e+T]=m[$*4+T]}return gn(d,[l,p,e],"int32")}function dd(s){return s!=null&&s.data instanceof Uint8Array}function fd(){return typeof window<"u"&&typeof ImageBitmap<"u"&&window.hasOwnProperty("createImageBitmap")}function yd(s){return s!=null&&s.width!==0&&s.height!==0}function gd(s){return fd()&&!(s instanceof ImageBitmap)&&yd(s)&&!dd(s)}async function bd(s,e=3){let t=null;if(Ue().getBool("WRAP_TO_IMAGEBITMAP")&&gd(s)){let n;try{n=await createImageBitmap(s,{premultiplyAlpha:"none"})}catch{n=null}n!=null&&n.width===s.width&&n.height===s.height?t=n:t=s}else t=s;return _n(t,e)}function vn(s){if(s.rank!==2&&s.rank!==3)throw new Error(`toPixels only supports rank 2 or 3 tensors, got rank ${s.rank}.`);const e=s.rank===2?1:s.shape[2];if(e>4||e===2)throw new Error(`toPixels only supports depth of size 1, 3 or 4 but got ${e}`);if(s.dtype!=="float32"&&s.dtype!=="int32")throw new Error(`Unsupported type for toPixels: ${s.dtype}. Please use float32 or int32 tensors.`)}function Nd(s){const e=(s==null?void 0:s.alpha)||1;if(e>1||e<0)throw new Error(`Alpha value ${e} is suppoed to be in range [0 - 1].`)}async function Td(s,e){let t=h(s,"img","toPixels");if(!(s instanceof _e)){const p=t;t=K(p,"int32"),p.dispose()}vn(t);const[n,r]=t.shape.slice(0,2),a=t.rank===2?1:t.shape[2],i=await t.data(),o=t.dtype==="float32"?255:1,u=new Uint8ClampedArray(r*n*4);for(let p=0;p<n*r;++p){const l=[0,0,0,255];for(let d=0;d<a;d++){const f=i[p*a+d];if(t.dtype==="float32"){if(f<0||f>1)throw new Error(`Tensor values for a float32 Tensor must be in the range [0 - 1] but encountered ${f}.`)}else if(t.dtype==="int32"&&(f<0||f>255))throw new Error(`Tensor values for a int32 Tensor must be in the range [0 - 255] but encountered ${f}.`);a===1?(l[0]=f*o,l[1]=f*o,l[2]=f*o):l[d]=f*o}const m=p*4;u[m+0]=Math.round(l[0]),u[m+1]=Math.round(l[1]),u[m+2]=Math.round(l[2]),u[m+3]=Math.round(l[3])}if(e!=null){xs||Us(Gs,b.backendName)!=null&&(console.warn("tf.browser.toPixels is not efficient to draw tensor on canvas. Please try tf.browser.draw instead."),xs=!0),e.width=r,e.height=n;const p=e.getContext("2d"),l=new ImageData(u,r,n);p.putImageData(l,0,0)}return t!==s&&t.dispose(),u}function wd(s,e,t){let n=h(s,"img","draw");if(!(s instanceof _e)){const i=n;n=K(i,"int32"),i.dispose()}vn(n),Nd(t==null?void 0:t.imageOptions);const r={image:n},a={canvas:e,options:t};b.runKernel(Gs,r,a)}const $d=g({fromPixels_:_n}),Jf=Object.freeze(Object.defineProperty({__proto__:null,draw:wd,fromPixels:$d,fromPixelsAsync:bd,toPixels:Td},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sd=Ue();Sd.registerFlag("KEEP_INTERMEDIATE_TENSORS",()=>!1,s=>{s&&console.warn("Keep intermediate tensors is ON. This will print the values of all intermediate tensors during model inference. Not all models support this mode. For details, check e2e/benchmarks/ model_config.js. This significantly impacts performance.")});/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * =============================================================================
 */var M;(function(s){s[s.DT_INVALID=0]="DT_INVALID",s[s.DT_FLOAT=1]="DT_FLOAT",s[s.DT_DOUBLE=2]="DT_DOUBLE",s[s.DT_INT32=3]="DT_INT32",s[s.DT_UINT8=4]="DT_UINT8",s[s.DT_INT16=5]="DT_INT16",s[s.DT_INT8=6]="DT_INT8",s[s.DT_STRING=7]="DT_STRING",s[s.DT_COMPLEX64=8]="DT_COMPLEX64",s[s.DT_INT64=9]="DT_INT64",s[s.DT_BOOL=10]="DT_BOOL",s[s.DT_QINT8=11]="DT_QINT8",s[s.DT_QUINT8=12]="DT_QUINT8",s[s.DT_QINT32=13]="DT_QINT32",s[s.DT_BFLOAT16=14]="DT_BFLOAT16",s[s.DT_QINT16=15]="DT_QINT16",s[s.DT_QUINT16=16]="DT_QUINT16",s[s.DT_UINT16=17]="DT_UINT16",s[s.DT_COMPLEX128=18]="DT_COMPLEX128",s[s.DT_HALF=19]="DT_HALF",s[s.DT_RESOURCE=20]="DT_RESOURCE",s[s.DT_VARIANT=21]="DT_VARIANT",s[s.DT_UINT32=22]="DT_UINT32",s[s.DT_UINT64=23]="DT_UINT64",s[s.DT_FLOAT_REF=101]="DT_FLOAT_REF",s[s.DT_DOUBLE_REF=102]="DT_DOUBLE_REF",s[s.DT_INT32_REF=103]="DT_INT32_REF",s[s.DT_UINT8_REF=104]="DT_UINT8_REF",s[s.DT_INT16_REF=105]="DT_INT16_REF",s[s.DT_INT8_REF=106]="DT_INT8_REF",s[s.DT_STRING_REF=107]="DT_STRING_REF",s[s.DT_COMPLEX64_REF=108]="DT_COMPLEX64_REF",s[s.DT_INT64_REF=109]="DT_INT64_REF",s[s.DT_BOOL_REF=110]="DT_BOOL_REF",s[s.DT_QINT8_REF=111]="DT_QINT8_REF",s[s.DT_QUINT8_REF=112]="DT_QUINT8_REF",s[s.DT_QINT32_REF=113]="DT_QINT32_REF",s[s.DT_BFLOAT16_REF=114]="DT_BFLOAT16_REF",s[s.DT_QINT16_REF=115]="DT_QINT16_REF",s[s.DT_QUINT16_REF=116]="DT_QUINT16_REF",s[s.DT_UINT16_REF=117]="DT_UINT16_REF",s[s.DT_COMPLEX128_REF=118]="DT_COMPLEX128_REF",s[s.DT_HALF_REF=119]="DT_HALF_REF",s[s.DT_RESOURCE_REF=120]="DT_RESOURCE_REF",s[s.DT_VARIANT_REF=121]="DT_VARIANT_REF",s[s.DT_UINT32_REF=122]="DT_UINT32_REF",s[s.DT_UINT64_REF=123]="DT_UINT64_REF"})(M||(M={}));var Os;(function(s){(function(e){e[e.LEGACY=0]="LEGACY",e[e.V1=1]="V1",e[e.V2=2]="V2"})(s.CheckpointFormatVersion||(s.CheckpointFormatVersion={}))})(Os||(Os={}));/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ms={};function Mf(s,e){const t={tfOpName:s,category:"custom",inputs:[],attrs:[],customExecutor:e};ms[s]=t}function kn(s){return ms[s]}function ey(s){delete ms[s]}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function c(s,e,t,n,r){const a=e.inputParams[s];if(a&&a.inputIndexStart!==void 0){const o=a.inputIndexStart,u=a.inputIndexEnd===0?void 0:a.inputIndexEnd===void 0?o+1:a.inputIndexEnd,p=o<0?e.inputNames.length+o:o;if(a.type==="tensor")return H(e.inputNames[p],t,n,r);if(a.type==="tensors"){const d=e.inputs.slice(o,u);return e.inputNames.slice(o,u).filter((N,$)=>{var T;return((T=d[$])===null||T===void 0?void 0:T.op)!=="NoOp"}).map(N=>H(N,t,n,r))}const l=H(e.inputNames[p],t,n,r),m=l.dataSync();return a.type==="number"?m[0]:Do(l.shape,m)}const i=e.attrParams[s];return i&&i.value}function H(s,e,t,n){const[r,a]=ee(s,t);if(n!=null){const o=n.getHashTableHandleByName(r);if(o!=null)return o}const i=t.currentContextIds.find(o=>!!e[gt(r,o)]);return i!==void 0?e[gt(r,i)][a]:void 0}function Is(s,e,t){return e[gt(s,t.currentContextId)]}function pe(s,e){const[t,n,r]=ee(s,e);return[gt(t,e&&e.currentContextId),n,r]}function gt(s,e){return e?`${s}-${e}`:s}function ee(s,e){if(s==="")return["",0,void 0];const t=e!=null&&e.parseNodeNameCache!=null;if(t){const a=e.parseNodeNameCache.get(s);if(a!=null)return a}const n=s.split(":");let r;if(n.length===1)r=[s,0,void 0];else{const a=n[0],i=n.length===3?n[1]:void 0,o=Number(n[n.length-1]);r=[a,o,i]}return t&&e.parseNodeNameCache.set(s,r),r}function ct(s,e,t){let n=c("pad",s,e,t);if(n==="explicit"){n=c("explicitPaddings",s,e,t);const r=[[0,0],[0,0],[0,0],[0,0]];for(let a=0;a<4;a++)r[a][0]=n[a*2],r[a][1]=n[a*2+1];return r}return n}function le(s){return s.kept?s:Ce(s)}/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _d=[{tfOpName:"Add",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddV2",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AddN",category:"arithmetic",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"BiasAdd",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"Sub",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"RealDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Div",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"DivNoNan",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorDiv",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mul",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Maximum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Minimum",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Pow",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SquaredDifference",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Mod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"FloorMod",category:"arithmetic",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],vd=Object.freeze(Object.defineProperty({__proto__:null,json:_d},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kd=[{tfOpName:"Abs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atan2",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Ceil",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ClipByValue",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"clipValueMin",type:"number"},{start:2,name:"clipValueMax",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Complex",category:"basic_math",inputs:[{start:0,name:"real",type:"tensor"},{start:1,name:"imag",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ComplexAbs",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cos",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Elu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Exp",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Floor",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Imag",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Neg",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Real",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"Tout",name:"outputType",type:"dtype",notSupported:!0}]},{tfOpName:"Prelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"alpha",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Relu6",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Selu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sigmoid",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sin",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Rsqrt",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Square",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Tanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Sign",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Round",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Expm1",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Log1p",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Reciprocal",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Softplus",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Asinh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Acosh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Atanh",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Erf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LeakyRelu",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"alpha",name:"alpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsNan",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsFinite",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"IsInf",category:"basic_math",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],Ed=Object.freeze(Object.defineProperty({__proto__:null,json:kd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xd=[{tfOpName:"EmptyTensorList",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"maxNumElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"LoopCond",category:"control",inputs:[{start:0,name:"pred",type:"tensor"}]},{tfOpName:"Switch",category:"control",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"pred",type:"tensor"}]},{tfOpName:"Merge",category:"control",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}]},{tfOpName:"Enter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"frame_name",name:"frameName",type:"string"},{tfName:"is_constant",name:"isConstant",type:"bool"}]},{tfOpName:"Exit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NextIteration",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayV3",category:"control",inputs:[{start:0,name:"size",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"dynamic_size",name:"dynamicSize",type:"bool"},{tfName:"clear_after_read",name:"clearAfterRead",type:"bool"},{tfName:"identical_element_shapes",name:"identicalElementShapes",type:"bool"},{tfName:"tensor_array_name",name:"name",type:"string"}]},{tfOpName:"TensorArrayWriteV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayReadV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"TensorArrayGatherV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape",name:"elementShape",type:"shape"}]},{tfOpName:"TensorArrayScatterV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"tensor",type:"tensor"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArrayConcatV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"element_shape_except0",name:"elementShapeExcept0",type:"shape",notSupported:!0}]},{tfOpName:"TensorArraySplitV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"tensor",type:"tensor"},{start:2,name:"lengths",type:"number[]"},{start:3,name:"flowIn",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"TensorArraySizeV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"},{start:1,name:"flowIn",type:"number"}]},{tfOpName:"TensorArrayCloseV3",category:"control",inputs:[{start:0,name:"tensorArrayId",type:"tensor"}]},{tfOpName:"StatelessIf",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"If",category:"control",inputs:[{start:0,name:"cond",type:"tensor"},{start:1,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"then_branch",name:"thenBranch",type:"func"},{tfName:"else_branch",name:"elseBranch",type:"func"}]},{tfOpName:"StatelessWhile",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"While",category:"control",inputs:[{start:0,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"cond",name:"cond",type:"func"},{tfName:"body",name:"body",type:"func"}]},{tfOpName:"TensorListScatter",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListScatterV2",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"},{start:3,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGather",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"indices",type:"number[]"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListGetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListSetItem",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"index",type:"number"},{start:2,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListReserve",category:"control",inputs:[{start:0,name:"elementShape",type:"shape"},{start:1,name:"numElements",type:"number"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListFromTensor",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListStack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"},{tfName:"num_elements",name:"numElements",type:"dtype"}]},{tfOpName:"TensorListSplit",category:"control",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"elementShape",type:"shape"},{start:2,name:"lengths",type:"number[]"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcat",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListConcatV2",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}],attrs:[{tfName:"element_shape",name:"elementShape",type:"shape"},{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPopBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"elementShape",type:"shape"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListPushBack",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"tensor",type:"tensor"}],attrs:[{tfName:"element_dtype",name:"elementDType",type:"dtype"}]},{tfOpName:"TensorListLength",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"}]},{tfOpName:"TensorListResize",category:"control",inputs:[{start:0,name:"tensorListId",type:"tensor"},{start:1,name:"size",type:"number"}]}],Od=Object.freeze(Object.defineProperty({__proto__:null,json:xd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Id=[{tfOpName:"AvgPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[],notSupported:!0},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPoolWithArgmax",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"include_batch_in_index",name:"includeBatchInIndex",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"AvgPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MaxPool3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"ksize",name:"kernelSize",type:"number[]"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Conv1D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"stride",name:"stride",type:"number"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NWC"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"dilation",name:"dilation",type:"number",defaultValue:1}]},{tfOpName:"Conv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"useCudnnOnGpu",name:"useCudnnOnGpu",type:"bool"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"_FusedConv2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"use_cudnn_on_gpu",name:"useCudnnOnGpu",type:"bool",defaultValue:!0},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2}]},{tfOpName:"Conv2DBackpropInput",category:"convolution",inputs:[{start:2,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:0,name:"outputShape",type:"number[]"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]",notSupported:!0}]},{tfOpName:"DepthwiseConv2d",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"DepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"FusedDepthwiseConv2dNative",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]",defaultValue:[1,1,1,1]},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"explicit_paddings",name:"explicitPaddings",type:"number[]",defaultValue:[]}]},{tfOpName:"Conv3D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"padding",name:"pad",type:"string"},{tfName:"data_format",name:"dataFormat",type:"string",defaultValue:"NHWC"},{tfName:"dilations",name:"dilations",type:"number[]"}]},{tfOpName:"Dilation2D",category:"convolution",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"filter",type:"tensor"}],attrs:[{tfName:"strides",name:"strides",type:"number[]"},{tfName:"rates",name:"dilations",type:"number[]"},{tfName:"padding",name:"pad",type:"string"}]}],Dd=Object.freeze(Object.defineProperty({__proto__:null,json:Id},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ad=[{tfOpName:"Fill",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"},{start:1,name:"value",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"LinSpace",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"num",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"OneHot",category:"creation",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"depth",type:"number"},{start:2,name:"onValue",type:"number",defaultValue:1},{start:3,name:"offValue",type:"number",defaultValue:0}],attrs:[{tfName:"axis",name:"axis",type:"number",notSupported:!0},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Ones",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"OnesLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"RandomStandardNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniform",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number",defaultValue:0},{tfName:"maxval",name:"maxval",type:"number",defaultValue:1},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"RandomUniformInt",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"minval",name:"minval",type:"number"},{tfName:"maxval",name:"maxval",type:"number"},{tfName:"seed",name:"seed",type:"number",defaultValue:0},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Range",category:"creation",inputs:[{start:0,name:"start",type:"number"},{start:1,name:"stop",type:"number"},{start:2,name:"step",type:"number",defaultValue:0}],attrs:[{tfName:"Tidx",name:"dtype",type:"dtype"}]},{tfOpName:"TruncatedNormal",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"means",name:"mean",type:"number",defaultValue:0},{tfName:"stddev",name:"stdDev",type:"number",defaultValue:1},{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number",defaultValue:0,notSupported:!0},{tfName:"dtype",name:"dtype",type:"dtype"},{tfName:"T",name:"T",type:"number",notSupported:!0}]},{tfOpName:"Zeros",category:"creation",inputs:[{start:0,name:"shape",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"ZerosLike",category:"creation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"Multinomial",category:"creation",inputs:[{start:0,name:"logits",type:"tensor"},{start:1,name:"numSamples",type:"number"}],attrs:[{tfName:"seed",name:"seed",type:"number"},{tfName:"seed2",name:"seed2",type:"number"},{tfName:"T",name:"dtype",type:"dtype"},{tfName:"output_dtype",name:"output_dtype",type:"dtype"}]}],Cd=Object.freeze(Object.defineProperty({__proto__:null,json:Ad},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zd=[{tfOpName:"NonMaxSuppressionV2",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV3",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}]},{tfOpName:"NonMaxSuppressionV4",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0},{tfName:"T_threshold",name:"threshold",type:"dtype",notSupported:!0},{tfName:"pad_to_max_output_size",name:"padToMaxOutputSize",type:"bool"}]},{tfOpName:"NonMaxSuppressionV5",category:"dynamic",inputs:[{start:0,name:"boxes",type:"tensor"},{start:1,name:"scores",type:"tensor"},{start:2,name:"maxOutputSize",type:"number"},{start:3,name:"iouThreshold",type:"number"},{start:4,name:"scoreThreshold",type:"number"},{start:5,name:"softNmsSigma",type:"number"}]},{tfOpName:"Where",category:"dynamic",inputs:[{start:0,name:"condition",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ListDiff",category:"dynamic",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]}],Pd=Object.freeze(Object.defineProperty({__proto__:null,json:zd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vd=[{tfOpName:"LowerBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"TopKV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"k",type:"number"}],attrs:[{tfName:"sorted",name:"sorted",type:"bool"}]},{tfOpName:"UpperBound",category:"evaluation",inputs:[{start:0,name:"sortedSequence",type:"tensor"},{start:1,name:"values",type:"tensor"}]},{tfOpName:"Unique",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"UniqueV2",category:"evaluation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]}],Ld=Object.freeze(Object.defineProperty({__proto__:null,json:Vd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Fd=[{tfOpName:"PlaceholderWithDefault",category:"graph",inputs:[{start:0,name:"default",type:"tensor"}],attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Placeholder",category:"graph",attrs:[{tfName:"shape",name:"shape",type:"shape"},{tfName:"dtype",name:"dtype",type:"dtype"}]},{tfOpName:"Const",category:"graph"},{tfOpName:"Identity",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IdentityN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Snapshot",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Rank",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Size",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"Shape",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"ShapeN",category:"graph",inputs:[{start:0,end:0,name:"x",type:"tensors"}]},{tfOpName:"Print",category:"graph",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"data",type:"tensors"}],attrs:[{tfName:"message",name:"message",type:"string"},{tfName:"first_n",name:"firstN",type:"number",notSupported:!0},{tfName:"summarize",name:"summarize",type:"number",defaultValue:3}]},{tfOpName:"NoOp",category:"graph",inputs:[]},{tfOpName:"StopGradient",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"FakeQuantWithMinMaxVars",category:"graph",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"min",name:"min",type:"number"},{tfName:"max",name:"max",type:"number"}]}],Rd=Object.freeze(Object.defineProperty({__proto__:null,json:Fd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Bd=[{tfOpName:"HashTable",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"HashTableV2",category:"hash_table",inputs:[],attrs:[{tfName:"shared_name",name:"sharedName",type:"string"},{tfName:"use_node_name_sharing",name:"useNodeNameSharing",type:"bool"},{tfName:"key_dtype",name:"keyDType",type:"dtype"},{tfName:"value_dtype",name:"valueDType",type:"dtype"}]},{tfOpName:"LookupTableImport",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableImportV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFind",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableFindV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"Tin",name:"tIn",type:"dtype",notSupported:!0},{tfName:"Tout",name:"tOut",type:"dtype",notSupported:!0}]},{tfOpName:"LookupTableSize",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"LookupTableSizeV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"}]},{tfOpName:"InitializeTable",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]},{tfOpName:"InitializeTableV2",category:"hash_table",inputs:[{start:0,name:"tableHandle",type:"tensor"},{start:1,name:"keys",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],jd=Object.freeze(Object.defineProperty({__proto__:null,json:Bd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const qd=[{tfOpName:"ResizeBilinear",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"ResizeNearestNeighbor",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"size",type:"number[]"}],attrs:[{tfName:"align_corners",name:"alignCorners",type:"bool"},{tfName:"half_pixel_centers",name:"halfPixelCenters",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"CropAndResize",category:"image",inputs:[{start:0,name:"image",type:"tensor"},{start:1,name:"boxes",type:"tensor"},{start:2,name:"boxInd",type:"tensor"},{start:3,name:"cropSize",type:"number[]"}],attrs:[{tfName:"method",name:"method",type:"string"},{tfName:"extrapolation_value",name:"extrapolationValue",type:"number"}]},{tfOpName:"ImageProjectiveTransformV3",category:"image",inputs:[{start:0,name:"images",type:"tensor"},{start:1,name:"transforms",type:"tensor"},{start:2,name:"outputShape",type:"number[]"},{start:3,name:"fillValue",type:"number"}],attrs:[{tfName:"interpolation",name:"interpolation",type:"string"},{tfName:"fill_mode",name:"fillMode",type:"string"}]}],Kd=Object.freeze(Object.defineProperty({__proto__:null,json:qd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Wd=[{tfOpName:"Equal",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"NotEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Greater",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"GreaterEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Less",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LessEqual",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalAnd",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalNot",category:"logical",inputs:[{start:0,name:"a",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"LogicalOr",category:"logical",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Select",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SelectV2",category:"logical",inputs:[{start:0,name:"condition",type:"tensor"},{start:1,name:"a",type:"tensor"},{start:2,name:"b",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BitwiseAnd",category:"logical",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"y",type:"tensor"}]}],Hd=Object.freeze(Object.defineProperty({__proto__:null,json:Wd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Gd=[{tfOpName:"_FusedMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"},{start:2,end:0,name:"args",type:"tensors"}],attrs:[{tfName:"num_args",name:"numArgs",type:"number"},{tfName:"fused_ops",name:"fusedOps",type:"string[]",defaultValue:[]},{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:1e-4},{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"leakyrelu_alpha",name:"leakyreluAlpha",type:"number",defaultValue:.2},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"MatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"transpose_a",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"transpose_b",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMul",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"BatchMatMulV2",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"b",type:"tensor"}],attrs:[{tfName:"adj_x",name:"transposeA",type:"bool",defaultValue:!1},{tfName:"adj_y",name:"transposeB",type:"bool",defaultValue:!1},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Transpose",category:"matrices",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"perm",type:"number[]"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Einsum",category:"matrices",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"equation",name:"equation",type:"string"},{tfName:"N",name:"n",type:"number",defaultValue:2},{tfName:"T",name:"dtype",type:"dtype"}]},{tfOpName:"MatrixBandPart",category:"matrices",inputs:[{start:0,name:"a",type:"tensor"},{start:1,name:"numLower",type:"tensor"},{start:1,name:"numUpper",type:"tensor"}]}],Ud=Object.freeze(Object.defineProperty({__proto__:null,json:Gd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Xd=[{tfOpName:"EuclideanNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool",defaultValue:!1}]},{tfOpName:"FusedBatchNorm",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV2",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"FusedBatchNormV3",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"scale",type:"tensor"},{start:2,name:"offset",type:"tensor"},{start:3,name:"mean",type:"tensor"},{start:4,name:"variance",type:"tensor"}],attrs:[{tfName:"epsilon",name:"epsilon",type:"number",defaultValue:.001},{tfName:"data_format",name:"dataFormat",type:"string",notSupported:!0}]},{tfOpName:"LRN",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"depth_radius",name:"radius",type:"number",defaultValue:5},{tfName:"bias",name:"bias",type:"number",defaultValue:1},{tfName:"alpha",name:"alpha",type:"number",defaultValue:1},{tfName:"beta",name:"beta",type:"number",defaultValue:.5}]},{tfOpName:"Softmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"LogSoftmax",category:"normalization",inputs:[{start:0,name:"x",type:"tensor"}]}],Yd=Object.freeze(Object.defineProperty({__proto__:null,json:Xd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Qd=[{tfOpName:"Bincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}]},{tfOpName:"DenseBincount",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"size",type:"number"},{start:2,name:"weights",type:"tensor"}],attrs:[{tfName:"binary_output",name:"binaryOutput",type:"bool"}]},{tfOpName:"Max",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Mean",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Min",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Sum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"All",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"Any",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"}]},{tfOpName:"ArgMax",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"ArgMin",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"Prod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}],attrs:[{tfName:"keep_dims",name:"keepDims",type:"bool"},{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"Cumprod",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]},{tfOpName:"Cumsum",category:"reduction",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}],attrs:[{tfName:"exclusive",name:"exclusive",type:"bool"},{tfName:"reverse",name:"reverse",type:"bool"}]}],Zd=Object.freeze(Object.defineProperty({__proto__:null,json:Qd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Jd=[{tfOpName:"ConcatV2",category:"slice_join",inputs:[{start:0,end:-1,name:"tensors",type:"tensors"},{start:-1,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"Concat",category:"slice_join",inputs:[{start:1,end:0,name:"tensors",type:"tensors"},{start:0,name:"axis",type:"number"}],attrs:[{tfName:"N",name:"n",type:"number",defaultValue:2}]},{tfOpName:"GatherV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"axis",type:"number",defaultValue:0}],attrs:[{tfName:"batch_dims",name:"batchDims",type:"number",defaultValue:0}]},{tfOpName:"Gather",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",notSupported:!0}]},{tfOpName:"Reverse",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"dims",type:"bool[]"}]},{tfOpName:"ReverseV2",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number[]"}]},{tfOpName:"Slice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"size",type:"number[]"}]},{tfOpName:"StridedSlice",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"begin",type:"number[]"},{start:2,name:"end",type:"number[]"},{start:3,name:"strides",type:"number[]"}],attrs:[{tfName:"begin_mask",name:"beginMask",type:"number",defaultValue:0},{tfName:"end_mask",name:"endMask",type:"number",defaultValue:0},{tfName:"new_axis_mask",name:"newAxisMask",type:"number",defaultValue:0},{tfName:"ellipsis_mask",name:"ellipsisMask",type:"number",defaultValue:0},{tfName:"shrink_axis_mask",name:"shrinkAxisMask",type:"number",defaultValue:0}]},{tfOpName:"Pack",category:"slice_join",inputs:[{start:0,end:0,name:"tensors",type:"tensors"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0}]},{tfOpName:"Unpack",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"}],attrs:[{tfName:"axis",name:"axis",type:"number",defaultValue:0},{tfName:"num",name:"num",type:"number",defaultValue:0,notSupported:!0}]},{tfOpName:"Tile",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"reps",type:"number[]"}]},{tfOpName:"Split",category:"slice_join",inputs:[{start:0,name:"axis",type:"number",defaultValue:0},{start:1,name:"x",type:"tensor"}],attrs:[{tfName:"num_split",name:"numOrSizeSplits",type:"number",defaultValue:1}]},{tfOpName:"SplitV",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"numOrSizeSplits",type:"number[]"},{start:2,name:"axis",type:"number",defaultValue:0}]},{tfOpName:"ScatterNd",category:"slice_join",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"shape",type:"number[]"}]},{tfOpName:"GatherNd",category:"slice_join",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"indices",type:"tensor"}]},{tfOpName:"SparseToDense",category:"slice_join",inputs:[{start:0,name:"sparseIndices",type:"tensor"},{start:1,name:"outputShape",type:"number[]"},{start:2,name:"sparseValues",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}],attrs:[{tfName:"validate_indices",name:"validateIndices",type:"bool",defaultValue:!1,notSupported:!0}]},{tfOpName:"TensorScatterUpdate",category:"slice_join",inputs:[{start:0,name:"tensor",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"values",type:"tensor"}]}],Md=Object.freeze(Object.defineProperty({__proto__:null,json:Jd},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const ef=[{tfOpName:"SparseFillEmptyRows",category:"sparse",inputs:[{start:0,name:"indices",type:"tensor"},{start:1,name:"values",type:"tensor"},{start:2,name:"denseShape",type:"tensor"},{start:3,name:"defaultValue",type:"tensor"}]},{tfOpName:"SparseReshape",category:"sparse",inputs:[{start:0,name:"inputIndices",type:"tensor"},{start:1,name:"inputShape",type:"tensor"},{start:2,name:"newShape",type:"tensor"}],attrs:[{tfName:"T",name:"dtype",type:"dtype",notSupported:!0}]},{tfOpName:"SparseSegmentMean",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]},{tfOpName:"SparseSegmentSum",category:"sparse",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"indices",type:"tensor"},{start:2,name:"segmentIds",type:"tensor"}]}],tf=Object.freeze(Object.defineProperty({__proto__:null,json:ef},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const sf=[{tfOpName:"FFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"IFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"}]},{tfOpName:"RFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]},{tfOpName:"IRFFT",category:"spectral",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"fft_length",type:"number",notSupported:!0}]}],nf=Object.freeze(Object.defineProperty({__proto__:null,json:sf},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const rf=[{tfOpName:"StaticRegexReplace",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"pattern",name:"pattern",type:"string"},{tfName:"rewrite",name:"rewrite",type:"string"},{tfName:"replace_global",name:"replaceGlobal",type:"bool"}]},{tfOpName:"StringNGrams",category:"string",inputs:[{start:0,name:"data",type:"tensor"},{start:1,name:"dataSplits",type:"tensor"}],attrs:[{tfName:"separator",name:"separator",type:"string"},{tfName:"ngram_widths",name:"nGramWidths",type:"number[]"},{tfName:"left_pad",name:"leftPad",type:"string"},{tfName:"right_pad",name:"rightPad",type:"string"},{tfName:"pad_width",name:"padWidth",type:"number"},{tfName:"preserve_short_sequences",name:"preserveShortSequences",type:"bool"}],outputs:["ngrams","ngrams_splits"]},{tfOpName:"StringSplit",category:"string",inputs:[{start:0,name:"input",type:"tensor"},{start:1,name:"delimiter",type:"tensor"}],attrs:[{tfName:"skip_empty",name:"skipEmpty",type:"bool"}],outputs:["indices","values","shape"]},{tfOpName:"StringToHashBucketFast",category:"string",inputs:[{start:0,name:"input",type:"tensor"}],attrs:[{tfName:"num_buckets",name:"numBuckets",type:"number"}]}],af=Object.freeze(Object.defineProperty({__proto__:null,json:rf},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2023 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const of=[{tfOpName:"Cast",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"SrcT",name:"sdtype",type:"dtype",notSupported:!0},{tfName:"DstT",name:"dtype",type:"dtype"}]},{tfOpName:"ExpandDims",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"axis",type:"number"}]},{tfOpName:"MirrorPad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"mode",name:"mode",type:"string"}]},{tfOpName:"Pad",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"}],attrs:[{tfName:"constant_value",name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"PadV2",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"padding",type:"number[]"},{start:2,name:"constantValue",type:"number",defaultValue:0}]},{tfOpName:"Reshape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"EnsureShape",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}]},{tfOpName:"Squeeze",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"axis",tfDeprecatedName:"squeeze_dims",name:"axis",type:"number[]"}]},{tfOpName:"SpaceToBatchND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"paddings",type:"number[]"}]},{tfOpName:"BatchToSpaceND",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"blockShape",type:"number[]"},{start:2,name:"crops",type:"number[]"}]},{tfOpName:"DepthToSpace",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"}],attrs:[{tfName:"block_size",name:"blockSize",type:"number"},{tfName:"data_format",name:"dataFormat",type:"string"}]},{tfOpName:"BroadcastTo",category:"transformation",inputs:[{start:0,name:"x",type:"tensor"},{start:1,name:"shape",type:"number[]"}],attrs:[]},{tfOpName:"BroadcastArgs",category:"transformation",inputs:[{start:0,name:"s0",type:"tensor"},{start:1,name:"s1",type:"tensor"}],attrs:[]}],uf=Object.freeze(Object.defineProperty({__proto__:null,json:of},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class Ds{static get Instance(){return this._instance||(this._instance=new this)}constructor(){const e=[vd,Ed,Od,Dd,Cd,Pd,Ld,Rd,jd,Kd,Hd,Ud,Yd,Zd,Md,tf,nf,af,uf],t=[].concat(...e.map(n=>n.json));this.opMappers=t.reduce((n,r)=>(n[r.tfOpName]=r,n),{})}transformGraph(e,t={}){const n=e.node,r=[],a=[],i=[],o=n.reduce(($,T)=>($[T.name]=this.mapNode(T),T.op.startsWith("Placeholder")?r.push($[T.name]):T.op==="Const"?a.push($[T.name]):(T.input==null||T.input.length===0)&&i.push($[T.name]),$),{});let u=[];const p=[];let l={},m={};t!=null&&(l=this.mapSignatureEntries(t.inputs),m=this.mapSignatureEntries(t.outputs));const d=Object.keys(o);d.forEach($=>{const T=o[$];T.inputNames.forEach((S,O)=>{const[C,,_]=pe(S),k=o[C];if(k.outputs!=null){const v=k.outputs.indexOf(_);if(v!==-1){const E=`${C}:${v}`;T.inputNames[O]=E}}T.inputs.push(k),k.children.push(T)})}),Object.keys(m).length===0?d.forEach($=>{const T=o[$];T.children.length===0&&p.push(T)}):Object.keys(m).forEach($=>{const[T]=pe($),S=o[T];S!=null&&(S.signatureKey=m[$],p.push(S))}),Object.keys(l).length>0?Object.keys(l).forEach($=>{const[T]=pe($),S=o[T];S&&(S.signatureKey=l[$],u.push(S))}):u=r;let f={};e.library!=null&&e.library.function!=null&&(f=e.library.function.reduce(($,T)=>($[T.signature.name]=this.mapFunction(T),$),{}));const N={nodes:o,inputs:u,outputs:p,weights:a,placeholders:r,signature:t,functions:f};return i.length>0&&(N.initNodes=i),N}mapSignatureEntries(e){return Object.keys(e||{}).reduce((t,n)=>(t[e[n].name]=n,t),{})}mapNode(e){const t=kn(e.op)||this.opMappers[e.op]||{};e.attr==null&&(e.attr={});const n={name:e.name,op:e.op,category:t.category,inputNames:(e.input||[]).map(r=>r.startsWith("^")?r.slice(1):r),inputs:[],children:[],inputParams:{},attrParams:{},rawAttrs:e.attr,outputs:t.outputs};return t.inputs!=null&&(n.inputParams=t.inputs.reduce((r,a)=>(r[a.name]={type:a.type,inputIndexStart:a.start,inputIndexEnd:a.end},r),{})),t.attrs!=null&&(n.attrParams=t.attrs.reduce((r,a)=>{const i=a.type;let o;switch(a.type){case"string":o=Lt(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=Lt(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"string[]":o=Wt(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=Wt(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"number":o=Rt(e.attr,a.tfName,a.defaultValue||0),o===void 0&&a.tfDeprecatedName&&(o=Rt(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"number[]":o=Kt(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=Kt(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool":o=Ft(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=Ft(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"bool[]":o=Gt(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=Gt(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape":o=qt(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=qt(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"shape[]":o=Ht(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=Ht(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype":o=Bt(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=Bt(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"dtype[]":o=jt(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=jt(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"func":o=As(e.attr,a.tfName,a.defaultValue),o===void 0&&a.tfDeprecatedName&&(o=As(e.attr,a.tfDeprecatedName,a.defaultValue));break;case"tensor":case"tensors":break;default:throw new Error(`Unsupported param type: ${a.type} for op: ${e.op}`)}return r[a.name]={value:o,type:i},r},{})),n}mapFunction(e){const t=e.nodeDef,n=[],r=[];let a={};t!=null&&(a=t.reduce((m,d)=>(m[d.name]=this.mapNode(d),d.op==="Const"&&r.push(m[d.name]),m),{}));const i=[],o=[];e.signature.inputArg.forEach(m=>{const[d]=pe(m.name),f={name:d,op:"Placeholder",inputs:[],inputNames:[],category:"graph",inputParams:{},attrParams:{dtype:{value:hs(m.type),type:"dtype"}},children:[]};f.signatureKey=m.name,i.push(f),a[d]=f}),Object.keys(a).forEach(m=>{const d=a[m];d.inputNames.forEach((f,N)=>{const[$,,T]=pe(f),S=a[$];if(S.outputs!=null){const O=S.outputs.indexOf(T);if(O!==-1){const C=`${$}:${O}`;d.inputNames[N]=C}}d.inputs.push(S),S.children.push(d)})});const p=e.ret;e.signature.outputArg.forEach(m=>{const[d,f]=pe(p[m.name]),N=a[d];N!=null&&(N.defaultOutput=f,o.push(N))});const l=this.mapArgsToSignature(e);return{nodes:a,inputs:i,outputs:o,weights:r,placeholders:n,signature:l}}mapArgsToSignature(e){return{methodName:e.signature.name,inputs:e.signature.inputArg.reduce((t,n)=>(t[n.name]=this.mapArgToTensorInfo(n),t),{}),outputs:e.signature.outputArg.reduce((t,n)=>(t[n.name]=this.mapArgToTensorInfo(n,e.ret),t),{})}}mapArgToTensorInfo(e,t){let n=e.name;return t!=null&&(n=t[n]),{name:n,dtype:e.type}}}function cf(s){const e=Ue().global;if(typeof e.atob<"u")return e.atob(s);if(typeof Buffer<"u")return new Buffer(s,"base64").toString();throw new Error("Unable to decode base64 in this environment. Missing built-in atob() or Buffer()")}function En(s,e){const t=Array.isArray(s)?String.fromCharCode.apply(null,s):cf(s);return e?t:t.toLowerCase()}function Lt(s,e,t,n=!1){const r=s[e];return r!=null?En(r.s,n):t}function Ft(s,e,t){const n=s[e];return n?n.b:t}function Rt(s,e,t){const n=s[e]||{},r=n.i!=null?n.i:n.f!=null?n.f:t;return typeof r=="number"?r:parseInt(r,10)}function hs(s){switch(typeof s=="string"&&(s=M[s]),s){case M.DT_FLOAT:case M.DT_HALF:return"float32";case M.DT_INT32:case M.DT_INT64:case M.DT_INT8:case M.DT_UINT8:return"int32";case M.DT_BOOL:return"bool";case M.DT_DOUBLE:return"float32";case M.DT_STRING:return"string";case M.DT_COMPLEX64:case M.DT_COMPLEX128:return"complex64";default:return null}}function As(s,e,t){const n=s[e];return n&&n.func?n.func.name:t}function Bt(s,e,t){const n=s[e];return n&&n.type?hs(n.type):t}function jt(s,e,t){const n=s[e];return n&&n.list&&n.list.type?n.list.type.map(r=>hs(r)):t}function xn(s){if(!s.unknownRank)return s.dim!=null?s.dim.map(e=>typeof e.size=="number"?e.size:parseInt(e.size,10)):[]}function qt(s,e,t){const n=s[e];return n&&n.shape?xn(n.shape):t}function Kt(s,e,t){const n=s[e];return n?((n.list.f&&n.list.f.length?n.list.f:n.list.i)||[]).map(r=>typeof r=="number"?r:parseInt(r,10)):t}function Wt(s,e,t,n=!1){const r=s[e];return r&&r.list&&r.list.s?r.list.s.map(a=>En(a,n)):t}function Ht(s,e,t){const n=s[e];return n&&n.list&&n.list.shape?n.list.shape.map(r=>xn(r)):t}function Gt(s,e,t){const n=s[e];return n&&n.list&&n.list.b?n.list.b:t}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class pf{constructor(e,t,n){this.node=e,this.tensorMap=t,this.context=n,this.inputs=[],this.attrs={},this.inputs=e.inputNames.map(r=>this.getInput(r)),e.rawAttrs!=null&&(this.attrs=Object.keys(e.rawAttrs).reduce((r,a)=>(r[a]=this.getAttr(a),r),{}))}getInput(e){return H(e,this.tensorMap,this.context)}getAttr(e,t){const n=this.node.rawAttrs[e];if(n.tensor!=null)return H(e,this.tensorMap,this.context);if(n.i!=null||n.f!=null)return Rt(this.node.rawAttrs,e,t);if(n.s!=null)return Lt(this.node.rawAttrs,e,t);if(n.b!=null)return Ft(this.node.rawAttrs,e,t);if(n.shape!=null)return qt(this.node.rawAttrs,e,t);if(n.type!=null)return Bt(this.node.rawAttrs,e,t);if(n.list!=null){if(n.list.i!=null||n.list.f!=null)return Kt(this.node.rawAttrs,e,t);if(n.list.s!=null)return Wt(this.node.rawAttrs,e,t);if(n.list.shape!=null)return Ht(this.node.rawAttrs,e,t);if(n.list.b!=null)return Gt(this.node.rawAttrs,e,t);if(n.list.type!=null)return jt(this.node.rawAttrs,e,t)}return t}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const U=Object.freeze(Object.defineProperty({__proto__:null,OP_SCOPE_SUFFIX:Ao,abs:se,acos:Ko,acosh:Ho,add:W,addN:Uo,all:Yo,any:Zo,argMax:Mo,argMin:ti,asin:ni,asinh:ai,atan:ii,atan2:ci,atanh:li,avgPool:Xs,avgPool3d:di,basicLSTMCell:Ti,batchNorm:St,batchNorm2d:vi,batchNorm3d:Ei,batchNorm4d:Oi,batchToSpaceND:Ys,bincount:Qs,bitwiseAnd:Ai,booleanMaskAsync:rm,broadcastArgs:zi,broadcastTo:tt,buffer:Fe,cast:K,ceil:Vi,clipByValue:Fi,clone:Ce,complex:Le,concat:Y,concat1d:Bi,concat2d:qi,concat3d:Wi,concat4d:Gi,conv1d:Yi,conv2d:_t,conv2dTranspose:Ji,conv3d:eu,conv3dTranspose:ru,cos:ou,cosh:uu,cosineWindow:ls,cumprod:pu,cumsum:mu,denseBincount:du,depthToSpace:yu,depthwiseConv2d:ss,diag:Nu,dilation2d:wu,div:X,divNoNan:vu,dot:Eu,dropout:gm,einsum:Ae,elu:Co,enclosingPowerOfTwo:Nn,ensureShape:Iu,equal:Js,erf:Au,euclideanNorm:Lu,exp:ke,expandDims:ye,expm1:ju,eye:en,fft:us,fill:Nt,floor:tn,floorDiv:zo,fused:Am,gather:sn,gatherND:dm,greater:kt,greaterEqual:nn,ifft:ft,imag:Et,image:ad,inTopKAsync:Nm,irfft:fn,isFinite:Qu,isInf:Ju,isNaN:ec,leakyRelu:Po,less:Pt,lessEqual:ns,linalg:od,linspace:nc,localResponseNormalization:ac,log:Xe,log1p:rn,logSigmoid:lc,logSoftmax:hc,logSumExp:on,logicalAnd:mt,logicalNot:un,logicalOr:cn,logicalXor:Nc,losses:id,lowerBound:wc,matMul:V,max:ze,maxPool:pn,maxPool3d:_c,maxPoolWithArgmax:kc,maximum:Vo,mean:ht,meshgrid:xc,min:zt,minimum:dt,mirrorPad:Dc,mod:Cc,moments:Pc,movingAverage:im,mul:I,multiRNNCell:Lc,multinomial:Rc,neg:ue,norm:vt,notEqual:ln,oneHot:qc,ones:$e,onesLike:Wc,op:g,outerProduct:Gc,pad:Je,pad1d:Yc,pad2d:Zc,pad3d:Mc,pad4d:tp,pool:op,pow:Qt,prelu:Lo,print:Fo,prod:up,raggedGather:pp,raggedRange:mp,raggedTensorToTensor:dp,rand:yp,randomGamma:Fp,randomNormal:hn,randomStandardNormal:jp,randomUniform:is,randomUniformInt:Wp,range:Ye,real:Qe,reciprocal:Up,relu:ts,relu6:Ro,reshape:w,reverse:Ee,reverse1d:Qp,reverse2d:Jp,reverse3d:el,reverse4d:sl,rfft:cs,round:dn,rsqrt:al,scalar:L,scatterND:cm,searchSorted:rs,selu:il,separableConv2d:cl,setdiff1dAsync:ll,sigmoid:We,sign:hl,signal:rd,sin:fl,sinh:gl,slice:F,slice1d:Nl,slice2d:wl,slice3d:Sl,slice4d:vl,softmax:El,softplus:an,spaceToBatchND:mn,sparse:ud,sparseToDense:mm,spectral:nd,split:Ze,sqrt:At,square:wt,squaredDifference:yn,squeeze:ps,stack:de,step:Bo,stridedSlice:Ll,string:cd,sub:P,sum:R,tan:Rl,tanh:Ct,tensor:ve,tensor1d:re,tensor2d:Ge,tensor3d:gn,tensor4d:Bl,tensor5d:jl,tensor6d:ql,tensorScatterUpdate:Wl,tile:He,topk:Gl,transpose:Vt,truncatedNormal:Xl,unique:Ql,unsortedSegmentSum:Jl,unstack:Oe,upperBound:em,variable:tm,where:ge,whereAsync:bn,zeros:$t,zerosLike:Xt},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lf=(s,e,t,n=U)=>{switch(s.op){case"BiasAdd":case"AddV2":case"Add":return[n.add(c("a",s,e,t),c("b",s,e,t))];case"AddN":return[n.addN(c("tensors",s,e,t))];case"FloorMod":case"Mod":return[n.mod(c("a",s,e,t),c("b",s,e,t))];case"Mul":return[n.mul(c("a",s,e,t),c("b",s,e,t))];case"RealDiv":case"Div":return[n.div(c("a",s,e,t),c("b",s,e,t))];case"DivNoNan":return[n.divNoNan(c("a",s,e,t),c("b",s,e,t))];case"FloorDiv":return[n.floorDiv(c("a",s,e,t),c("b",s,e,t))];case"Sub":return[n.sub(c("a",s,e,t),c("b",s,e,t))];case"Minimum":return[n.minimum(c("a",s,e,t),c("b",s,e,t))];case"Maximum":return[n.maximum(c("a",s,e,t),c("b",s,e,t))];case"Pow":return[n.pow(c("a",s,e,t),c("b",s,e,t))];case"SquaredDifference":return[n.squaredDifference(c("a",s,e,t),c("b",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const mf=(s,e,t,n=U)=>{switch(s.op){case"Abs":case"ComplexAbs":return[n.abs(c("x",s,e,t))];case"Acos":return[n.acos(c("x",s,e,t))];case"Acosh":return[n.acosh(c("x",s,e,t))];case"Asin":return[n.asin(c("x",s,e,t))];case"Asinh":return[n.asinh(c("x",s,e,t))];case"Atan":return[n.atan(c("x",s,e,t))];case"Atan2":return[n.atan2(c("x",s,e,t),c("y",s,e,t))];case"Atanh":return[n.atanh(c("x",s,e,t))];case"Ceil":return[n.ceil(c("x",s,e,t))];case"Complex":return[n.complex(c("real",s,e,t),c("imag",s,e,t))];case"Cos":return[n.cos(c("x",s,e,t))];case"Cosh":return[n.cosh(c("x",s,e,t))];case"Elu":return[n.elu(c("x",s,e,t))];case"Erf":return[n.erf(c("x",s,e,t))];case"Exp":return[n.exp(c("x",s,e,t))];case"Expm1":return[n.expm1(c("x",s,e,t))];case"Floor":return[n.floor(c("x",s,e,t))];case"Log":return[n.log(c("x",s,e,t))];case"Log1p":return[n.log1p(c("x",s,e,t))];case"Imag":return[n.imag(c("x",s,e,t))];case"Neg":return[n.neg(c("x",s,e,t))];case"Reciprocal":return[n.reciprocal(c("x",s,e,t))];case"Real":return[n.real(c("x",s,e,t))];case"Relu":return[n.relu(c("x",s,e,t))];case"Round":return[n.round(c("x",s,e,t))];case"Selu":return[n.selu(c("x",s,e,t))];case"Sigmoid":return[n.sigmoid(c("x",s,e,t))];case"Sin":return[n.sin(c("x",s,e,t))];case"Sign":return[n.sign(c("x",s,e,t))];case"Sinh":return[n.sinh(c("x",s,e,t))];case"Softplus":return[n.softplus(c("x",s,e,t))];case"Sqrt":return[n.sqrt(c("x",s,e,t))];case"Square":return[n.square(c("x",s,e,t))];case"Tanh":return[n.tanh(c("x",s,e,t))];case"Tan":return[n.tan(c("x",s,e,t))];case"ClipByValue":return[n.clipByValue(c("x",s,e,t),c("clipValueMin",s,e,t),c("clipValueMax",s,e,t))];case"Relu6":return[n.relu6(c("x",s,e,t))];case"Rsqrt":return[n.rsqrt(H(s.inputNames[0],e,t))];case"LeakyRelu":return[n.leakyRelu(c("x",s,e,t),c("alpha",s,e,t))];case"Prelu":return[n.prelu(c("x",s,e,t),c("alpha",s,e,t))];case"IsNan":return[n.isNaN(H(s.inputNames[0],e,t))];case"IsInf":return[n.isInf(H(s.inputNames[0],e,t))];case"IsFinite":return[n.isFinite(H(s.inputNames[0],e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function ne(s,e,t=""){if(!(typeof s=="number"||typeof e=="number")){y(s.length===e.length,()=>t+` Shapes ${s} and ${e} must match`);for(let n=0;n<s.length;n++){const r=s[n],a=e[n];y(r<0||a<0||r===a,()=>t+` Shapes ${s} and ${e} must match`)}}}function Cs(s){return!(typeof s=="number"||s.some(e=>e<0))}function Ke(s,e,t){let n=Ut(s,t);const r=!Cs(n);if(r&&e.length===0)throw new Error(`Tried to calculate elements of an empty list with non-fully-defined elementShape: ${n}`);if(r&&e.forEach(a=>{n=Ut(a.shape,n)}),!Cs(n))throw new Error(`Non-fully-defined elementShape: ${n}`);return n}function Ut(s,e){if(typeof s=="number")return e;if(typeof e=="number")return s;if(s.length!==e.length)throw new Error(`Incompatible ranks during merge: ${s} vs. ${e}`);const t=[];for(let n=0;n<s.length;++n){const r=s[n],a=e[n];if(r>=0&&a>=0&&r!==a)throw new Error(`Incompatible shape during merge: ${s} vs. ${e}`);t[n]=r>=0?r:a}return t}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class hf{constructor(e,t,n,r,a,i,o){this.name=e,this.dtype=t,this.maxSize=n,this.elementShape=r,this.identicalElementShapes=a,this.dynamicSize=i,this.clearAfterRead=o,this.tensors=[],this.closed_=!1,this.idTensor=L(0),me(this.idTensor)}get id(){return this.idTensor.id}get closed(){return this.closed_}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.tensor.id))&&t.tensor.dispose()}),this.tensors=[],this.closed_=!0,this.idTensor.dispose()}size(){return this.tensors.length}read(e){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||e>=this.size())throw new Error(`Tried to read from index ${e}, but array size is: ${this.size()}`);const t=this.tensors[e];if(t.cleared)throw new Error(`TensorArray ${this.name}: Could not read index ${e} twice because it was cleared after a previous read (perhaps try setting clear_after_read = false?).`);return this.clearAfterRead&&(t.cleared=!0),t.read=!0,t.tensor}readMany(e){return e.map(t=>this.read(t))}write(e,t){if(this.closed_)throw new Error(`TensorArray ${this.name} has already been closed.`);if(e<0||!this.dynamicSize&&e>=this.maxSize)throw new Error(`Tried to write to index ${e}, but array is not resizeable and size is: ${this.maxSize}`);const n=this.tensors[e]||{};if(t.dtype!==this.dtype)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e},
          because the value dtype is ${t.dtype}, but TensorArray dtype is ${this.dtype}.`);if(this.size()===0&&(this.elementShape==null||this.elementShape.length===0)&&(this.elementShape=t.shape),ne(this.elementShape,t.shape,`TensorArray ${this.name}: Could not write to TensorArray index ${e}.`),n.read)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been read.`);if(n.written)throw new Error(`TensorArray ${this.name}: Could not write to TensorArray index ${e}, because it has already been written.`);n.tensor=t,me(t),n.written=!0,this.tensors[e]=n}writeMany(e,t){if(e.length!==t.length)throw new Error(`TensorArray ${this.name}: could not write multiple tensors,because the index size: ${e.length} is not the same as tensors size: ${t.length}.`);e.forEach((n,r)=>this.write(n,t[r]))}gather(e,t){if(t&&t!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but gather requested dtype ${t}`);if(e)e=e.slice(0,this.size());else{e=[];for(let r=0;r<this.size();r++)e.push(r)}if(e.length===0)return ve([],[0].concat(this.elementShape));const n=this.readMany(e);return ne(this.elementShape,n[0].shape,"TensorArray shape mismatch: "),de(n,0)}concat(e){if(e&&e!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but concat requested dtype ${e}`);if(this.size()===0)return ve([],[0].concat(this.elementShape));const t=[];for(let r=0;r<this.size();r++)t.push(r);const n=this.readMany(t);return ne(this.elementShape,n[0].shape,`TensorArray shape mismatch: tensor array shape (${this.elementShape}) vs first tensor shape (${n[0].shape})`),Y(n,0)}scatter(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);if(e.length!==t.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${t.shape[0]}`);const n=Math.max(...e);if(!this.dynamicSize&&n>=this.maxSize)throw new Error(`Max index must be < array size (${n}  vs. ${this.maxSize})`);this.writeMany(e,Oe(t,0))}split(e,t){if(t.dtype!==this.dtype)throw new Error(`TensorArray dtype is ${this.dtype} but tensor has dtype ${t.dtype}`);let n=0;const r=e.map(u=>(n+=u,n));if(n!==t.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${n}, and tensor's shape is: ${t.shape}`);if(!this.dynamicSize&&e.length!==this.maxSize)throw new Error(`TensorArray's size is not equal to the size of lengths (${this.maxSize} vs. ${e.length}), and the TensorArray is not marked as dynamically resizeable`);const a=n===0?0:t.size/n,i=[];ae(()=>{t=w(t,[1,n,a]);for(let u=0;u<e.length;++u){const l=[0,u===0?0:r[u-1],0],m=[1,e[u],a];i[u]=w(F(t,l,m),this.elementShape)}return i});const o=[];for(let u=0;u<e.length;u++)o[u]=u;this.writeMany(o,i)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class xe{get id(){return this.idTensor.id}constructor(e,t,n,r=-1){this.tensors=e,this.elementShape=t,this.elementDtype=n,e!=null&&e.forEach(a=>{if(n!==a.dtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${a.dtype}`);ne(t,a.shape,"TensorList shape mismatch: "),me(a)}),this.idTensor=L(0),this.maxNumElements=r,me(this.idTensor)}copy(){return new xe([...this.tensors],this.elementShape,this.elementDtype)}clearAndClose(e){this.tensors.forEach(t=>{(e==null||!e.has(t.id))&&t.dispose()}),this.tensors.length=0,this.idTensor.dispose()}size(){return this.tensors.length}stack(e,t,n=-1){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(n!==-1&&this.tensors.length!==n)throw new Error(`Operation expected a list with ${n} elements but got a list with ${this.tensors.length} elements.`);ne(e,this.elementShape,"TensorList shape mismatch: ");const r=Ke(this.elementShape,this.tensors,e);return ae(()=>{const a=this.tensors.map(i=>w(i,r));return de(a,0)})}popBack(e,t){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);if(this.size()===0)throw new Error("Trying to pop from an empty list.");const n=Ke(this.elementShape,this.tensors,e),r=this.tensors.pop();return r.kept=!1,ne(r.shape,e,"TensorList shape mismatch: "),w(r,n)}pushBack(e){if(e.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${e.dtype}, but list elements ${this.elementDtype}`);if(ne(e.shape,this.elementShape,"TensorList shape mismatch: "),this.maxNumElements===this.size())throw new Error("Trying to push element into a full list.");me(e),this.tensors.push(e)}resize(e){if(e<0)throw new Error(`TensorListResize expects size to be non-negative. Got: ${e}`);if(this.maxNumElements!==-1&&e>this.maxNumElements)throw new Error(`TensorListResize input size ${e} is greater maxNumElement ${this.maxNumElements}.`);const t=new xe([],this.elementShape,this.elementDtype,this.maxNumElements);t.tensors.length=e;for(let n=0;n<Math.min(this.tensors.length,e);++n)t.tensors[n]=this.tensors[n];return t}getItem(e,t,n){if(n!==this.elementDtype)throw new Error(`Invalid data types; op elements ${n}, but list elements ${this.elementDtype}`);if(e<0||e>this.tensors.length)throw new Error(`Trying to access element ${e} in a list with ${this.tensors.length} elements.`);if(this.tensors[e]==null)throw new Error(`element at index ${e} is null.`);ne(this.tensors[e].shape,t,"TensorList shape mismatch: ");const r=Ke(this.elementShape,this.tensors,t);return w(this.tensors[e],r)}setItem(e,t){if(t.dtype!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t.dtype}, but list elements ${this.elementDtype}`);if(e<0||this.maxNumElements!==-1&&e>=this.maxNumElements)throw new Error(`Trying to set element ${e} in a list with max ${this.maxNumElements} elements.`);ne(this.elementShape,t.shape,"TensorList shape mismatch: "),me(t),this.tensors[e]!=null&&(this.tensors[e].kept=!1),this.tensors[e]=t}gather(e,t,n){if(t!==this.elementDtype)throw new Error(`Invalid data types; op elements ${t}, but list elements ${this.elementDtype}`);ne(this.elementShape,n,"TensorList shape mismatch: "),e=e.slice(0,this.size());const r=Ke(this.elementShape,this.tensors,n);return e.length===0?ve([],[0].concat(r)):ae(()=>{const a=e.map(i=>w(this.tensors[i],r));return de(a,0)})}concat(e,t){if(e&&e!==this.elementDtype)throw new Error(`TensorList dtype is ${this.elementDtype} but concat requested dtype ${e}`);ne(this.elementShape,t,"TensorList shape mismatch: ");const n=Ke(this.elementShape,this.tensors,t);return this.size()===0?ve([],[0].concat(n)):ae(()=>{const r=this.tensors.map(a=>w(a,n));return Y(r,0)})}}function df(s,e,t){const n=s.dtype;if(s.shape.length<1)throw new Error(`Tensor must be at least a vector, but saw shape: ${s.shape}`);if(s.dtype!==t)throw new Error(`Invalid data types; op elements ${s.dtype}, but list elements ${t}`);const r=s.shape.slice(1);ne(r,e,"TensorList shape mismatch: ");const a=Oe(s);return new xe(a,e,n)}function ff(s,e,t,n){return new xe([],s,e,n)}function yf(s,e,t,n){if(e.length!==s.shape[0])throw new Error(`Expected len(indices) == tensor.shape[0], but saw: ${e.length} vs. ${s.shape[0]}`);const r=Math.max(...e);if(n!=null&&n!==-1&&r>=n)throw new Error(`Max index must be < array size (${r}  vs. ${n})`);const a=new xe([],t,s.dtype,n),i=Oe(s,0);return e.forEach((o,u)=>{a.setItem(o,i[u])}),a}function gf(s,e,t){let n=0;const r=e.map(l=>(n+=l,n));if(n!==s.shape[0])throw new Error(`Expected sum of lengths to be equal to
          tensor.shape[0], but sum of lengths is
        ${n}, and tensor's shape is: ${s.shape}`);const a=s.shape.slice(1),i=Ut(a,t),o=n===0?0:s.size/n,u=ae(()=>{const l=[];s=w(s,[1,n,o]);for(let m=0;m<e.length;++m){const f=[0,m===0?0:r[m-1],0],N=[1,e[m],o];l[m]=w(F(s,f,N),i)}return s.dispose(),l}),p=new xe([],t,s.dtype,e.length);for(let l=0;l<u.length;l++)p.setItem(l,u[l]);return p}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const bf=async(s,e,t)=>{switch(s.op){case"If":case"StatelessIf":{const n=c("thenBranch",s,e,t),r=c("elseBranch",s,e,t),a=c("cond",s,e,t),i=c("args",s,e,t);return(await a.data())[0]?t.functionMap[n].executeFunctionAsync(i,t.tensorArrayMap,t.tensorListMap):t.functionMap[r].executeFunctionAsync(i,t.tensorArrayMap,t.tensorListMap)}case"While":case"StatelessWhile":{const n=c("body",s,e,t),r=c("cond",s,e,t),a=c("args",s,e,t),i=await t.functionMap[r].executeFunctionAsync(a,t.tensorArrayMap,t.tensorListMap),o=a.map(l=>l.id);let u=await i[0].data();i.forEach(l=>{!l.kept&&o.indexOf(l.id)===-1&&l.dispose()});let p=a;for(;u[0];){const l=p;p=await t.functionMap[n].executeFunctionAsync(p,t.tensorArrayMap,t.tensorListMap);const m=p.map(f=>f.id);l.forEach(f=>{!f.kept&&o.indexOf(f.id)===-1&&m.indexOf(f.id)===-1&&f.dispose()});const d=await t.functionMap[r].executeFunctionAsync(p,t.tensorArrayMap,t.tensorListMap);u=await d[0].data(),d.forEach(f=>{!f.kept&&o.indexOf(f.id)===-1&&m.indexOf(f.id)===-1&&f.dispose()})}return p}case"LoopCond":{const n=c("pred",s,e,t);return[le(n)]}case"Switch":{const n=c("pred",s,e,t);let r=c("data",s,e,t);return r.kept||(r=le(r)),(await n.data())[0]?[void 0,r]:[r,void 0]}case"Merge":{const n=s.inputNames.find(r=>H(r,e,t)!==void 0);if(n){const r=H(n,e,t);return[le(r)]}return}case"Enter":{const n=c("frameName",s,e,t),r=c("tensor",s,e,t);return t.enterFrame(n),[le(r)]}case"Exit":{const n=c("tensor",s,e,t);return t.exitFrame(),[le(n)]}case"NextIteration":{const n=c("tensor",s,e,t);return t.nextIteration(),[le(n)]}case"TensorArrayV3":{const n=c("size",s,e,t),r=c("dtype",s,e,t),a=c("elementShape",s,e,t),i=c("dynamicSize",s,e,t),o=c("clearAfterRead",s,e,t),u=c("identicalElementShapes",s,e,t),p=c("name",s,e,t),l=new hf(p,r,n,a,u,i,o);return t.addTensorArray(l),[l.idTensor,L(1)]}case"TensorArrayWriteV3":{const n=c("tensorArrayId",s,e,t),r=c("index",s,e,t),a=c("tensor",s,e,t),i=t.getTensorArray(n.id);return i.write(r,a),[i.idTensor]}case"TensorArrayReadV3":{const n=c("tensorArrayId",s,e,t),r=c("index",s,e,t);return[t.getTensorArray(n.id).read(r)]}case"TensorArrayGatherV3":{const n=c("tensorArrayId",s,e,t),r=c("indices",s,e,t),a=c("dtype",s,e,t);return[t.getTensorArray(n.id).gather(r,a)]}case"TensorArrayScatterV3":{const n=c("tensorArrayId",s,e,t),r=c("indices",s,e,t),a=c("tensor",s,e,t),i=t.getTensorArray(n.id);return i.scatter(r,a),[i.idTensor]}case"TensorArrayConcatV3":{const n=c("tensorArrayId",s,e,t),r=t.getTensorArray(n.id),a=c("dtype",s,e,t);return[r.concat(a)]}case"TensorArraySplitV3":{const n=c("tensorArrayId",s,e,t),r=c("tensor",s,e,t),a=c("lengths",s,e,t),i=t.getTensorArray(n.id);return i.split(a,r),[i.idTensor]}case"TensorArraySizeV3":{const n=c("tensorArrayId",s,e,t),r=t.getTensorArray(n.id);return[L(r.size(),"int32")]}case"TensorArrayCloseV3":{const n=c("tensorArrayId",s,e,t),r=t.getTensorArray(n.id);return r.clearAndClose(),[r.idTensor]}case"TensorListSetItem":{const n=c("tensorListId",s,e,t),r=c("index",s,e,t),a=c("tensor",s,e,t),i=t.getTensorList(n.id);return i.setItem(r,a),[i.idTensor]}case"TensorListGetItem":{const n=c("tensorListId",s,e,t),r=c("index",s,e,t),a=c("elementShape",s,e,t),i=c("elementDType",s,e,t);return[t.getTensorList(n.id).getItem(r,a,i)]}case"TensorListScatterV2":case"TensorListScatter":{const n=c("indices",s,e,t),r=c("tensor",s,e,t),a=c("elementShape",s,e,t),i=c("numElements",s,e,t),o=yf(r,n,a,i);return t.addTensorList(o),[o.idTensor]}case"TensorListReserve":case"EmptyTensorList":{const n=c("elementShape",s,e,t),r=c("elementDType",s,e,t);let a;s.op==="TensorListReserve"?a="numElements":a="maxNumElements";const i=c(a,s,e,t),o=s.op==="TensorListReserve"?-1:i,u=ff(n,r,i,o);return t.addTensorList(u),[u.idTensor]}case"TensorListGather":{const n=c("tensorListId",s,e,t),r=c("indices",s,e,t),a=c("elementShape",s,e,t),i=c("elementDType",s,e,t);return[t.getTensorList(n.id).gather(r,i,a)]}case"TensorListStack":{const n=c("tensorListId",s,e,t),r=c("elementShape",s,e,t),a=c("elementDType",s,e,t),i=c("numElements",s,e,t);return[t.getTensorList(n.id).stack(r,a,i)]}case"TensorListFromTensor":{const n=c("tensor",s,e,t),r=c("elementShape",s,e,t),a=c("elementDType",s,e,t),i=df(n,r,a);return t.addTensorList(i),[i.idTensor]}case"TensorListConcat":case"TensorListConcatV2":{const n=c("tensorListId",s,e,t),r=t.getTensorList(n.id),a=c("dtype",s,e,t),i=c("elementShape",s,e,t);return[r.concat(a,i)]}case"TensorListPushBack":{const n=c("tensorListId",s,e,t),r=c("tensor",s,e,t),a=t.getTensorList(n.id);return a.pushBack(r),[a.idTensor]}case"TensorListPopBack":{const n=c("tensorListId",s,e,t),r=c("elementShape",s,e,t),a=c("elementDType",s,e,t);return[t.getTensorList(n.id).popBack(r,a)]}case"TensorListSplit":{const n=c("tensor",s,e,t),r=c("elementShape",s,e,t),a=c("lengths",s,e,t),i=gf(n,a,r);return t.addTensorList(i),[i.idTensor]}case"TensorListLength":{const n=c("tensorListId",s,e,t),r=t.getTensorList(n.id);return[L(r.size(),"int32")]}case"TensorListResize":{const n=c("tensorListId",s,e,t),r=c("size",s,e,t),i=t.getTensorList(n.id).resize(r);return t.addTensorList(i),[i.idTensor]}default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function zs(s,e,t){const[n,r]=c("fusedOps",s,e,t),a=n==="biasadd",i=!a,o=r==="prelu",u=n==="fusedbatchnorm",p=c("numArgs",s,e,t);if(a){if(o&&p!==2)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!o&&a&&p!==1)throw new Error("FusedConv2d and DepthwiseConv2d with BiasAdd must have one extra argument: bias.")}if(u)throw new Error("FusedConv2d and DepthwiseConv2d with FusedBatchNorm is not supported");const l=c("strides",s,e,t),m=ct(s,e,t),d=c("dataFormat",s,e,t).toUpperCase(),f=c("dilations",s,e,t);let[N,$]=c("args",s,e,t);i&&($=N,N=void 0);const T=c("leakyreluAlpha",s,e,t);return{stride:l,pad:m,dataFormat:d,dilations:f,biasArg:N,preluArg:$,activationFunc:r,leakyreluAlpha:T}}const Nf=(s,e,t,n=U)=>{switch(s.op){case"Conv1D":{const r=c("stride",s,e,t),a=c("pad",s,e,t),i=c("dataFormat",s,e,t).toUpperCase(),o=c("dilation",s,e,t);return[n.conv1d(c("x",s,e,t),c("filter",s,e,t),r,a,i,o)]}case"Conv2D":{const r=c("strides",s,e,t),a=ct(s,e,t),i=c("dataFormat",s,e,t).toUpperCase(),o=c("dilations",s,e,t);return[n.conv2d(c("x",s,e,t),c("filter",s,e,t),[r[1],r[2]],a,i,[o[1],o[2]])]}case"_FusedConv2D":{const{stride:r,pad:a,dataFormat:i,dilations:o,biasArg:u,preluArg:p,activationFunc:l,leakyreluAlpha:m}=zs(s,e,t);return[n.fused.conv2d({x:c("x",s,e,t),filter:c("filter",s,e,t),strides:[r[1],r[2]],pad:a,dataFormat:i,dilations:[o[1],o[2]],bias:u,activation:l,preluActivationWeights:p,leakyreluAlpha:m})]}case"FusedDepthwiseConv2dNative":{const{stride:r,pad:a,dataFormat:i,dilations:o,biasArg:u,preluArg:p,activationFunc:l,leakyreluAlpha:m}=zs(s,e,t);return[n.fused.depthwiseConv2d({x:c("x",s,e,t),filter:c("filter",s,e,t),strides:[r[1],r[2]],pad:a,dataFormat:i,dilations:[o[1],o[2]],bias:u,activation:l,preluActivationWeights:p,leakyreluAlpha:m})]}case"Conv2DBackpropInput":case"Conv2dTranspose":{const r=c("outputShape",s,e,t),a=c("strides",s,e,t),i=ct(s,e,t);return[n.conv2dTranspose(c("x",s,e,t),c("filter",s,e,t),r,[a[1],a[2]],i)]}case"DepthwiseConv2dNative":case"DepthwiseConv2d":{const r=c("strides",s,e,t),a=ct(s,e,t),i=c("dilations",s,e,t),o=c("dataFormat",s,e,t).toUpperCase();return[n.depthwiseConv2d(c("input",s,e,t),c("filter",s,e,t),[r[1],r[2]],a,o,[i[1],i[2]])]}case"Conv3D":{const r=c("strides",s,e,t),a=c("pad",s,e,t),i=c("dataFormat",s,e,t).toUpperCase(),o=c("dilations",s,e,t);return[n.conv3d(c("x",s,e,t),c("filter",s,e,t),[r[1],r[2],r[3]],a,i,[o[1],o[2],o[3]])]}case"AvgPool":{const r=c("strides",s,e,t),a=c("pad",s,e,t),i=c("kernelSize",s,e,t);return[n.avgPool(c("x",s,e,t),[i[1],i[2]],[r[1],r[2]],a)]}case"MaxPool":{const r=c("strides",s,e,t),a=c("pad",s,e,t),i=c("kernelSize",s,e,t);return[n.maxPool(c("x",s,e,t),[i[1],i[2]],[r[1],r[2]],a)]}case"MaxPoolWithArgmax":{const r=c("strides",s,e,t),a=c("pad",s,e,t),i=c("kernelSize",s,e,t),o=c("includeBatchInIndex",s,e,t),{result:u,indexes:p}=n.maxPoolWithArgmax(c("x",s,e,t),[i[1],i[2]],[r[1],r[2]],a,o);return[u,p]}case"AvgPool3D":{const r=c("strides",s,e,t),a=c("pad",s,e,t),i=c("kernelSize",s,e,t);return[n.avgPool3d(c("x",s,e,t),[i[1],i[2],i[3]],[r[1],r[2],r[3]],a)]}case"MaxPool3D":{const r=c("strides",s,e,t),a=c("pad",s,e,t),i=c("kernelSize",s,e,t);return[n.maxPool3d(c("x",s,e,t),[i[1],i[2],i[3]],[r[1],r[2],r[3]],a)]}case"Dilation2D":{const r=c("strides",s,e,t),a=c("pad",s,e,t),i=c("dilations",s,e,t),o=r[1],u=r[2],p=i[1],l=i[2];return[n.dilation2d(c("x",s,e,t),c("filter",s,e,t),[o,u],a,[p,l],"NHWC")]}default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Tf=(s,e,t,n=U)=>{switch(s.op){case"Fill":{const r=c("shape",s,e,t),a=c("dtype",s,e,t),i=c("value",s,e,t);return[n.fill(r,i,a)]}case"LinSpace":{const r=c("start",s,e,t),a=c("stop",s,e,t),i=c("num",s,e,t);return[n.linspace(r,a,i)]}case"Multinomial":{const r=c("logits",s,e,t),a=c("numSamples",s,e,t),i=c("seed",s,e,t);return[n.multinomial(r,a,i)]}case"OneHot":{const r=c("indices",s,e,t),a=c("depth",s,e,t),i=c("onValue",s,e,t),o=c("offValue",s,e,t),u=c("dtype",s,e,t);return[n.oneHot(r,a,i,o,u)]}case"Ones":return[n.ones(c("shape",s,e,t),c("dtype",s,e,t))];case"OnesLike":return[n.onesLike(c("x",s,e,t))];case"RandomStandardNormal":return[n.randomStandardNormal(c("shape",s,e,t),c("dtype",s,e,t),c("seed",s,e,t))];case"RandomUniform":return[n.randomUniform(c("shape",s,e,t),c("minval",s,e,t),c("maxval",s,e,t),c("dtype",s,e,t))];case"RandomUniformInt":return[n.randomUniformInt(c("shape",s,e,t),c("minval",s,e,t),c("maxval",s,e,t),c("seed",s,e,t))];case"Range":{const r=c("start",s,e,t),a=c("stop",s,e,t),i=c("step",s,e,t);return[n.range(r,a,i,c("dtype",s,e,t))]}case"TruncatedNormal":{const r=c("shape",s,e,t),a=c("mean",s,e,t),i=c("stdDev",s,e,t),o=c("seed",s,e,t);return[n.truncatedNormal(r,a,i,c("dtype",s,e,t),o)]}case"Zeros":return[n.zeros(c("shape",s,e,t),c("dtype",s,e,t))];case"ZerosLike":return[n.zerosLike(c("x",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Dt(s,e,t){const n=c("boxes",s,e,t),r=c("scores",s,e,t),a=c("maxOutputSize",s,e,t),i=c("iouThreshold",s,e,t),o=c("scoreThreshold",s,e,t),u=c("softNmsSigma",s,e,t);return{boxes:n,scores:r,maxOutputSize:a,iouThreshold:i,scoreThreshold:o,softNmsSigma:u}}const wf=async(s,e,t,n,r=U)=>{switch(s.op){case"NonMaxSuppressionV5":{const{boxes:a,scores:i,maxOutputSize:o,iouThreshold:u,scoreThreshold:p,softNmsSigma:l}=Dt(s,e,t),m=await r.image.nonMaxSuppressionWithScoreAsync(a,i,o,u,p,l);return[m.selectedIndices,m.selectedScores]}case"NonMaxSuppressionV4":{const{boxes:a,scores:i,maxOutputSize:o,iouThreshold:u,scoreThreshold:p}=Dt(s,e,t),l=c("padToMaxOutputSize",s,e,t),m=await r.image.nonMaxSuppressionPaddedAsync(a,i,o,u,p,l);return[m.selectedIndices,m.validOutputs]}case"NonMaxSuppressionV3":case"NonMaxSuppressionV2":{const{boxes:a,scores:i,maxOutputSize:o,iouThreshold:u,scoreThreshold:p}=Dt(s,e,t);return[await r.image.nonMaxSuppressionAsync(a,i,o,u,p)]}case"Where":{const a=r.cast(c("condition",s,e,t),"bool"),i=[await r.whereAsync(a)];return a.dispose(),i}case"ListDiff":return r.setdiff1dAsync(c("x",s,e,t),c("y",s,e,t));default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const $f=(s,e,t,n=U)=>{switch(s.op){case"LowerBound":{const r=c("sortedSequence",s,e,t),a=c("values",s,e,t);return[n.lowerBound(r,a)]}case"TopKV2":{const r=c("x",s,e,t),a=c("k",s,e,t),i=c("sorted",s,e,t),o=n.topk(r,a,i);return[o.values,o.indices]}case"UpperBound":{const r=c("sortedSequence",s,e,t),a=c("values",s,e,t);return[n.upperBound(r,a)]}case"Unique":{const r=c("x",s,e,t),a=n.unique(r);return[a.values,a.indices]}case"UniqueV2":{const r=c("x",s,e,t),a=c("axis",s,e,t),i=n.unique(r,a);return[i.values,i.indices]}default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Sf=(s,e,t,n=U)=>{switch(s.op){case"Const":return e[s.name];case"PlaceholderWithDefault":const r=c("default",s,e,t);return[H(s.name,e,t)||r];case"Placeholder":return[H(s.name,e,t)];case"Identity":case"StopGradient":case"FakeQuantWithMinMaxVars":{const l=c("x",s,e,t);return[le(l)]}case"IdentityN":return c("x",s,e,t).map(l=>le(l));case"Snapshot":const a=c("x",s,e,t);return[le(a)];case"Shape":return[n.tensor1d(c("x",s,e,t).shape,"int32")];case"ShapeN":return c("x",s,e,t).map(l=>n.tensor1d(l.shape));case"Size":return[n.scalar(c("x",s,e,t).size,"int32")];case"Rank":return[n.scalar(c("x",s,e,t).rank,"int32")];case"NoOp":return[n.scalar(1)];case"Print":const i=c("x",s,e,t),o=c("data",s,e,t),u=c("message",s,e,t),p=c("summarize",s,e,t);console.warn("The graph has a tf.print() operation,usually used for debugging, which slows down performance."),console.log(u);for(let l=0;l<o.length;l++)console.log(Array.prototype.slice.call(o[l].dataSync()).slice(0,p));return[i];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class _f{get id(){return this.handle.id}constructor(e,t){this.keyDType=e,this.valueDType=t,this.handle=L(0),this.tensorMap=new Map,me(this.handle)}clearAndClose(){this.tensorMap.forEach(e=>e.dispose()),this.tensorMap.clear(),this.handle.dispose()}size(){return this.tensorMap.size}tensorSize(){return L(this.size(),"int32")}async import(e,t){this.checkKeyAndValueTensor(e,t);const n=await e.data();return this.tensorMap.forEach(r=>r.dispose()),this.tensorMap.clear(),ae(()=>{const r=Oe(t),a=n.length,i=r.length;y(a===i,()=>`The number of elements doesn't match, keys has ${a} elements, the values has ${i} elements.`);for(let o=0;o<a;o++){const u=n[o],p=r[o];me(p),this.tensorMap.set(u,p)}return this.handle})}async find(e,t){this.checkKeyAndValueTensor(e,t);const n=await e.data();return ae(()=>{const r=[];for(let a=0;a<n.length;a++){const i=n[a],o=this.findWithDefault(i,t);r.push(o)}return de(r)})}findWithDefault(e,t){const n=this.tensorMap.get(e);return n??t}checkKeyAndValueTensor(e,t){if(e.dtype!==this.keyDType)throw new Error(`Expect key dtype ${this.keyDType}, but got ${e.dtype}`);if(t.dtype!==this.valueDType)throw new Error(`Expect value dtype ${this.valueDType}, but got ${t.dtype}`)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const vf=async(s,e,t,n)=>{switch(s.op){case"HashTable":case"HashTableV2":{const r=n.getHashTableHandleByName(s.name);if(r!=null)return[r];{const a=c("keyDType",s,e,t),i=c("valueDType",s,e,t),o=new _f(a,i);return n.addHashTable(s.name,o),[o.handle]}}case"InitializeTable":case"InitializeTableV2":case"LookupTableImport":case"LookupTableImportV2":{const r=c("tableHandle",s,e,t,n),a=c("keys",s,e,t),i=c("values",s,e,t);return[await n.getHashTableById(r.id).import(a,i)]}case"LookupTableFind":case"LookupTableFindV2":{const r=c("tableHandle",s,e,t,n),a=c("keys",s,e,t),i=c("defaultValue",s,e,t);return[await n.getHashTableById(r.id).find(a,i)]}case"LookupTableSize":case"LookupTableSizeV2":{const r=c("tableHandle",s,e,t,n);return[n.getHashTableById(r.id).tensorSize()]}default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const kf=(s,e,t,n=U)=>{switch(s.op){case"ResizeBilinear":{const r=c("images",s,e,t),a=c("size",s,e,t),i=c("alignCorners",s,e,t),o=c("halfPixelCenters",s,e,t);return[n.image.resizeBilinear(r,[a[0],a[1]],i,o)]}case"ResizeNearestNeighbor":{const r=c("images",s,e,t),a=c("size",s,e,t),i=c("alignCorners",s,e,t),o=c("halfPixelCenters",s,e,t);return[n.image.resizeNearestNeighbor(r,[a[0],a[1]],i,o)]}case"CropAndResize":{const r=c("image",s,e,t),a=c("boxes",s,e,t),i=c("boxInd",s,e,t),o=c("cropSize",s,e,t),u=c("method",s,e,t),p=c("extrapolationValue",s,e,t);return[n.image.cropAndResize(r,a,i,o,u,p)]}case"ImageProjectiveTransformV3":{const r=c("images",s,e,t),a=c("transforms",s,e,t),i=c("outputShape",s,e,t),o=c("fillValue",s,e,t),u=c("interpolation",s,e,t),p=c("fillMode",s,e,t);return[n.image.transform(r,a,u.toLowerCase(),p.toLowerCase(),o,i)]}default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ef=(s,e,t,n=U)=>{switch(s.op){case"Equal":return[n.equal(c("a",s,e,t),c("b",s,e,t))];case"NotEqual":return[n.notEqual(c("a",s,e,t),c("b",s,e,t))];case"Greater":return[n.greater(c("a",s,e,t),c("b",s,e,t))];case"GreaterEqual":return[n.greaterEqual(c("a",s,e,t),c("b",s,e,t))];case"Less":return[n.less(c("a",s,e,t),c("b",s,e,t))];case"LessEqual":return[n.lessEqual(c("a",s,e,t),c("b",s,e,t))];case"LogicalAnd":return[n.logicalAnd(c("a",s,e,t),c("b",s,e,t))];case"LogicalNot":return[n.logicalNot(c("a",s,e,t))];case"LogicalOr":return[n.logicalOr(c("a",s,e,t),c("b",s,e,t))];case"Select":case"SelectV2":return[n.where(c("condition",s,e,t),c("a",s,e,t),c("b",s,e,t))];case"BitwiseAnd":return[n.bitwiseAnd(c("a",s,e,t),c("b",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const xf=(s,e,t,n=U)=>{switch(s.op){case"BatchMatMul":case"BatchMatMulV2":case"MatMul":return[n.matMul(c("a",s,e,t),c("b",s,e,t),c("transposeA",s,e,t),c("transposeB",s,e,t))];case"Einsum":return[n.einsum(c("equation",s,e,t),...c("tensors",s,e,t))];case"Transpose":return[n.transpose(c("x",s,e,t),c("perm",s,e,t))];case"_FusedMatMul":const[r,a]=c("fusedOps",s,e,t),i=r==="biasadd",o=a==="prelu",u=c("numArgs",s,e,t),p=c("leakyreluAlpha",s,e,t);if(i){if(o&&u!==2)throw new Error("Fused MatMul with BiasAdd and Prelu must have two extra arguments: bias and alpha.");if(!o&&u!==1)throw new Error("Fused MatMul with BiasAdd must have one extra argument: bias.")}const[l,m]=c("args",s,e,t);return[n.fused.matMul({a:c("a",s,e,t),b:c("b",s,e,t),transposeA:c("transposeA",s,e,t),transposeB:c("transposeB",s,e,t),bias:l,activation:a,preluActivationWeights:m,leakyreluAlpha:p})];case"MatrixBandPart":return[n.linalg.bandPart(c("a",s,e,t),c("numLower",s,e,t),c("numUpper",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Of=(s,e,t,n=U)=>{switch(s.op){case"EuclideanNorm":return[n.euclideanNorm(c("x",s,e,t),c("axis",s,e,t),c("keepDims",s,e,t))];case"FusedBatchNorm":case"FusedBatchNormV2":return[n.batchNorm(c("x",s,e,t),c("mean",s,e,t),c("variance",s,e,t),c("offset",s,e,t),c("scale",s,e,t),c("epsilon",s,e,t))];case"FusedBatchNormV3":return[n.batchNorm(c("x",s,e,t),c("mean",s,e,t),c("variance",s,e,t),c("offset",s,e,t),c("scale",s,e,t),c("epsilon",s,e,t))];case"LRN":return[n.localResponseNormalization(c("x",s,e,t),c("radius",s,e,t),c("bias",s,e,t),c("alpha",s,e,t),c("beta",s,e,t))];case"Softmax":return[n.softmax(c("x",s,e,t))];case"LogSoftmax":return[n.logSoftmax(c("x",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2022 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const If=(s,e,t,n=U)=>{switch(s.op){case"RaggedGather":{const{outputNestedSplits:r,outputDenseValues:a}=n.raggedGather(c("paramsNestedSplits",s,e,t),c("paramsDenseValues",s,e,t),c("indices",s,e,t),c("outputRaggedRank",s,e,t));return r.concat(a)}case"RaggedRange":{const{rtNestedSplits:r,rtDenseValues:a}=n.raggedRange(c("starts",s,e,t),c("limits",s,e,t),c("splits",s,e,t));return[r,a]}case"RaggedTensorToTensor":return[n.raggedTensorToTensor(c("shape",s,e,t),c("values",s,e,t),c("defaultValue",s,e,t),c("rowPartitionTensors",s,e,t),c("rowPartitionTypes",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Df=(s,e,t,n=U)=>{switch(s.op){case"Max":{const o=c("axis",s,e,t),u=c("keepDims",s,e,t);return[n.max(c("x",s,e,t),o,u)]}case"Mean":{const o=c("axis",s,e,t),u=c("keepDims",s,e,t);return[n.mean(c("x",s,e,t),o,u)]}case"Min":{const o=c("axis",s,e,t),u=c("keepDims",s,e,t);return[n.min(c("x",s,e,t),o,u)]}case"Sum":{const o=c("axis",s,e,t),u=c("keepDims",s,e,t);return[n.sum(c("x",s,e,t),o,u)]}case"All":{const o=c("axis",s,e,t),u=c("keepDims",s,e,t);return[n.all(c("x",s,e,t),o,u)]}case"Any":{const o=c("axis",s,e,t),u=c("keepDims",s,e,t);return[n.any(c("x",s,e,t),o,u)]}case"ArgMax":{const o=c("axis",s,e,t);return[n.argMax(c("x",s,e,t),o)]}case"ArgMin":{const o=c("axis",s,e,t);return[n.argMin(c("x",s,e,t),o)]}case"Prod":{const o=c("axis",s,e,t),u=c("keepDims",s,e,t);return[n.prod(c("x",s,e,t),o,u)]}case"Cumprod":{const o=c("axis",s,e,t),u=c("exclusive",s,e,t),p=c("reverse",s,e,t);return[n.cumprod(c("x",s,e,t),o,u,p)]}case"Cumsum":{const o=c("axis",s,e,t),u=c("exclusive",s,e,t),p=c("reverse",s,e,t);return[n.cumsum(c("x",s,e,t),o,u,p)]}case"Bincount":const r=c("x",s,e,t),a=c("weights",s,e,t),i=c("size",s,e,t);return[n.bincount(r,a,i)];case"DenseBincount":{const o=c("x",s,e,t),u=c("weights",s,e,t),p=c("size",s,e,t),l=c("binaryOutput",s,e,t);return[n.denseBincount(o,u,p,l)]}default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Af=(s,e,t,n=U)=>{switch(s.op){case"ConcatV2":case"Concat":{const r=c("n",s,e,t),a=c("axis",s,e,t);let i=c("tensors",s,e,t);return i=i.slice(0,r),[n.concat(i,a)]}case"Gather":{const r=c("x",s,e,t),a=c("indices",s,e,t);return[n.gather(r,n.cast(a,"int32"),0)]}case"GatherV2":{const r=c("axis",s,e,t),a=c("batchDims",s,e,t),i=c("x",s,e,t),o=c("indices",s,e,t);return[n.gather(i,n.cast(o,"int32"),r,a)]}case"Reverse":{const r=c("dims",s,e,t),a=[];for(let o=0;o<r.length;o++)r[o]&&a.push(o);const i=c("x",s,e,t);return[n.reverse(i,a)]}case"ReverseV2":{const r=c("axis",s,e,t),a=c("x",s,e,t);return[n.reverse(a,r)]}case"Slice":{const r=c("begin",s,e,t),a=c("size",s,e,t);return[n.slice(c("x",s,e,t),r,a)]}case"StridedSlice":{const r=c("begin",s,e,t),a=c("end",s,e,t),i=c("strides",s,e,t),o=c("beginMask",s,e,t),u=c("endMask",s,e,t),p=c("ellipsisMask",s,e,t),l=c("newAxisMask",s,e,t),m=c("shrinkAxisMask",s,e,t),d=c("x",s,e,t);return[n.stridedSlice(d,r,a,i,o,u,p,l,m)]}case"Pack":return ae(()=>{const r=c("axis",s,e,t),a=c("tensors",s,e,t),i=a[0].shape,o=n.squeeze(a[0]).shape,u=a.map(p=>{const l=Pe(p.shape,i);if(!l&&!Pe(n.squeeze(p).shape,o))throw new Error("the input tensors shape does not match");return l?p:n.reshape(p,i)});return[n.stack(u,r)]});case"Unpack":{const r=c("axis",s,e,t),a=c("tensor",s,e,t);return n.unstack(a,r)}case"Tile":{const r=c("reps",s,e,t);return[n.tile(c("x",s,e,t),r)]}case"Split":case"SplitV":{const r=c("axis",s,e,t),a=c("numOrSizeSplits",s,e,t),i=c("x",s,e,t);return n.split(i,a,r)}case"ScatterNd":{const r=c("indices",s,e,t),a=c("values",s,e,t),i=c("shape",s,e,t);return[n.scatterND(r,a,i)]}case"GatherNd":{const r=c("x",s,e,t),a=c("indices",s,e,t);return[n.gatherND(r,a)]}case"SparseToDense":{const r=c("sparseIndices",s,e,t),a=c("outputShape",s,e,t),i=c("sparseValues",s,e,t),o=c("defaultValue",s,e,t);return[n.sparseToDense(r,i,a,i.dtype===o.dtype?o:n.cast(o,i.dtype))]}case"TensorScatterUpdate":{const r=c("indices",s,e,t),a=c("values",s,e,t),i=c("tensor",s,e,t);return[n.tensorScatterUpdate(i,r,a)]}default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cf=(s,e,t,n=U)=>{switch(s.op){case"SparseFillEmptyRows":{const{outputIndices:r,outputValues:a,emptyRowIndicator:i,reverseIndexMap:o}=n.sparse.sparseFillEmptyRows(c("indices",s,e,t),c("values",s,e,t),c("denseShape",s,e,t),c("defaultValue",s,e,t));return[r,a,i,o]}case"SparseReshape":{const{outputIndices:r,outputShape:a}=n.sparse.sparseReshape(c("inputIndices",s,e,t),c("inputShape",s,e,t),c("newShape",s,e,t));return[r,a]}case"SparseSegmentMean":return[n.sparse.sparseSegmentMean(c("data",s,e,t),c("indices",s,e,t),c("segmentIds",s,e,t))];case"SparseSegmentSum":return[n.sparse.sparseSegmentSum(c("data",s,e,t),c("indices",s,e,t),c("segmentIds",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const zf=(s,e,t,n=U)=>{switch(s.op){case"FFT":return[n.fft(c("x",s,e,t))];case"IFFT":return[n.ifft(c("x",s,e,t))];case"RFFT":return[n.rfft(c("x",s,e,t))];case"IRFFT":return[n.irfft(c("x",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2021 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Pf=(s,e,t,n=U)=>{switch(s.op){case"StaticRegexReplace":return[n.string.staticRegexReplace(c("input",s,e,t),c("pattern",s,e,t),c("rewrite",s,e,t),c("replaceGlobal",s,e,t))];case"StringNGrams":{const{nGrams:r,nGramsSplits:a}=n.string.stringNGrams(c("data",s,e,t),c("dataSplits",s,e,t),c("separator",s,e,t),c("nGramWidths",s,e,t),c("leftPad",s,e,t),c("rightPad",s,e,t),c("padWidth",s,e,t),c("preserveShortSequences",s,e,t));return[r,a]}case"StringSplit":{const{indices:r,values:a,shape:i}=n.string.stringSplit(c("input",s,e,t),c("delimiter",s,e,t),c("skipEmpty",s,e,t));return[r,a,i]}case"StringToHashBucketFast":return[n.string.stringToHashBucketFast(c("input",s,e,t),c("numBuckets",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Vf=(s,e,t,n=U)=>{switch(s.op){case"Cast":return[n.cast(c("x",s,e,t),c("dtype",s,e,t))];case"ExpandDims":{const r=c("axis",s,e,t);return[n.expandDims(c("x",s,e,t),r)]}case"Squeeze":{const r=c("axis",s,e,t);return[n.squeeze(c("x",s,e,t),r)]}case"Reshape":return[n.reshape(c("x",s,e,t),c("shape",s,e,t))];case"EnsureShape":return[n.ensureShape(c("x",s,e,t),c("shape",s,e,t))];case"MirrorPad":return[n.mirrorPad(c("x",s,e,t),c("padding",s,e,t),c("mode",s,e,t))];case"PadV2":case"Pad":return[n.pad(c("x",s,e,t),c("padding",s,e,t),c("constantValue",s,e,t))];case"SpaceToBatchND":{const r=c("blockShape",s,e,t),a=c("paddings",s,e,t);return[n.spaceToBatchND(c("x",s,e,t),r,a)]}case"BatchToSpaceND":{const r=c("blockShape",s,e,t),a=c("crops",s,e,t);return[n.batchToSpaceND(c("x",s,e,t),r,a)]}case"DepthToSpace":{const r=c("blockSize",s,e,t),a=c("dataFormat",s,e,t).toUpperCase();return[n.depthToSpace(c("x",s,e,t),r,a)]}case"BroadcastTo":return[n.broadcastTo(c("x",s,e,t),c("shape",s,e,t))];case"BroadcastArgs":return[n.broadcastArgs(c("s0",s,e,t),c("s1",s,e,t))];default:throw TypeError(`Node type ${s.op} is not implemented`)}};/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ps(s,e,t,n,r=ae){const a=((i,o,u)=>{switch(i.category){case"arithmetic":return r(()=>lf(i,o,u));case"basic_math":return r(()=>mf(i,o,u));case"control":return bf(i,o,u);case"convolution":return r(()=>Nf(i,o,u));case"creation":return r(()=>Tf(i,o,u));case"dynamic":return wf(i,o,u);case"evaluation":return r(()=>$f(i,o,u));case"image":return r(()=>kf(i,o,u));case"graph":return r(()=>Sf(i,o,u));case"logical":return r(()=>Ef(i,o,u));case"matrices":return r(()=>xf(i,o,u));case"normalization":return r(()=>Of(i,o,u));case"ragged":return r(()=>If(i,o,u));case"reduction":return r(()=>Df(i,o,u));case"slice_join":return r(()=>Af(i,o,u));case"sparse":return r(()=>Cf(i,o,u));case"spectral":return r(()=>zf(i,o,u));case"string":return r(()=>Pf(i,o,u));case"transformation":return r(()=>Vf(i,o,u));case"hash_table":return vf(i,o,u,n);case"custom":const p=kn(i.op);if(p&&p.customExecutor)return p.customExecutor(new pf(i,o,u));throw TypeError(`Custom op ${i.op} is not registered.`);default:throw TypeError(`Unknown op '${i.op}'. File an issue at https://github.com/tensorflow/tfjs/issues so we can add it, or register a custom execution with tf.registerOp()`)}})(s,e,t);return lt(a)?a.then(i=>[].concat(i)):[].concat(a)}class Vs{constructor(e={},t={},n={},r={},a){this.weightMap=e,this.tensorArrayMap=t,this.tensorListMap=n,this.functionMap=r,this.parseNodeNameCache=a,this.rootContext={id:0,frameName:"",iterationId:0},this.contexts=[this.rootContext],this.lastId=0,this.generateCurrentContextIds()}newFrame(e,t){return{id:e,frameName:t,iterationId:0}}set currentContext(e){this.contexts!==e&&(this.contexts=e,this.generateCurrentContextIds())}get currentContext(){return this.contexts}get currentContextId(){return this._currentContextIds[0]}get currentContextIds(){return this._currentContextIds}generateCurrentContextIds(){const e=[];for(let t=0;t<this.contexts.length-1;t++){const n=this.contexts.slice(0,this.contexts.length-t);e.push(this.contextIdforContexts(n))}e.push(""),this._currentContextIds=e}contextIdforContexts(e){return e?e.map(t=>t.id===0&&t.iterationId===0?"":`${t.frameName}-${t.iterationId}`).join("/"):""}enterFrame(e){this.contexts&&(this.lastId++,this.contexts=this.contexts.slice(),this.contexts.push(this.newFrame(this.lastId,e)),this._currentContextIds.unshift(this.contextIdforContexts(this.contexts)))}exitFrame(){if(this.contexts&&this.contexts.length>1)this.contexts=this.contexts.slice(),this.contexts.splice(-1),this.currentContextIds.shift();else throw new Error("Cannot exit frame, the context is empty")}nextIteration(){if(this.contexts&&this.contexts.length>0){this.contexts=this.contexts.slice(),this.lastId++;const e=Object.assign({},this.contexts[this.contexts.length-1]);e.iterationId+=1,e.id=this.lastId,this.contexts.splice(-1,1,e),this._currentContextIds.splice(0,1,this.contextIdforContexts(this.contexts))}else throw new Error("Cannot increase frame iteration, the context is empty")}getWeight(e){return this.weightMap[e]}addTensorArray(e){this.tensorArrayMap[e.id]=e}getTensorArray(e){return this.tensorArrayMap[e]}addTensorList(e){this.tensorListMap[e.id]=e}getTensorList(e){return this.tensorListMap[e]}dispose(e){for(const t in this.tensorArrayMap)this.tensorArrayMap[t].clearAndClose(e);for(const t in this.tensorListMap)this.tensorListMap[t].clearAndClose(e)}}/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ls(s,e,t,n){const r=new Set,a=[];let i=null,o=null;const u=new Set,p=new Set(Object.keys(s).map(d=>ee(d)[0]));n=n||[];const l=new Set(n.map(d=>ee(d.name)[0])),m=[...e];for(;m.length>0;){const d=m.pop();if((we(d)||Wf(d)||Hf(d))&&i==null&&(i=d,o=i.children.map(f=>f.name).filter(f=>r.has(f))),r.add(d.name),t[d.name]==null&&!p.has(d.name)&&!l.has(d.name)){if(d.inputs.length===0){a.push(d.name);continue}d.inputs.forEach(f=>{u.has(f.name)||(u.add(f.name),m.push(f))})}}return{inputs:s,outputs:e,usedNodes:r,missingInputs:a,dynamicNode:i,syncInputs:o}}function Lf(s,e){const{usedNodes:t,inputs:n}=e,r=Object.keys(n).map(T=>ee(T)[0]).map(T=>s.nodes[T]),a=s.initNodes||[],i=T=>t.has(typeof T=="string"?T:T.name);function o(T){return[...new Map(T.map(S=>[S.name,S])).values()]}const u=o([...r,...s.weights,...a]).filter(i),p=o([...u,...Object.values(s.nodes)]).filter(i),l=new Map(p.map(T=>[T.name,T])),m={};for(const T of p){m[T.name]=m[T.name]||0;for(const S of T.children)i(S)||(m[S.name]=Number.POSITIVE_INFINITY),m[S.name]=(m[S.name]||0)+1}const d=Object.entries(m).filter(([,T])=>T===0).map(([T])=>T),f=[...d];for(;d.length>0;){const T=d.pop(),S=l.get(T);for(const O of S.children.filter(i))--m[O.name]===0&&(f.push(O.name),d.push(O.name))}const N=f.map(T=>l.get(T)),$=Ff(N,u);return Rf($,u),$}function Ff(s,e){const t=new Map(s.map(i=>[i.name,i])),n=e.map(i=>i.name),r=new Set(n);for(;n.length>0;){const i=n.pop(),o=t.get(i);for(const u of o.children)!t.has(u.name)||r.has(u.name)||(r.add(u.name),n.push(u.name))}return s.filter(i=>r.has(i.name))}class et extends Error{constructor(e){super(`NodesExecutionOrderError: ${e}`)}}function Rf(s,e){const t=new Map(s.map((o,u)=>[o.name,u])),n=new Set(e.map(o=>o.name)),r=o=>n.has(typeof o=="string"?o:o.name),a=new Set(s.map(o=>o.name)),i=o=>a.has(typeof o=="string"?o:o.name);for(const o of s){for(const u of o.children.filter(i)){if(!t.has(u.name))throw new et(`Child ${u.name} of node ${o.name} is unreachable.`);if(t.get(o.name)>t.get(u.name))throw new et(`Node ${o.name} is scheduled to run after its child ${u.name}.`)}if(!r(o))for(const u of o.inputs){if(!t.has(u.name))throw new et(`Input ${u.name} of node ${o.name} is unreachable.`);if(t.get(u.name)>t.get(o.name))throw new et(`Node ${o.name} is scheduled to run before its input ${u.name}.`)}}}function Bf(s){const e=new Map(s.map((o,u)=>[o.name,u])),t=Number.MAX_SAFE_INTEGER,n=s.map((o,u)=>we(o)?t:u),r=o=>{const u=n[e.get(o.name)];return u??-1},a=s.map((o,u)=>o.children.map(r).reduce((p,l)=>Math.max(p,l),n[u])),i=new Map;for(let o=0;o<s.length;++o){const u=a[o];if(u===t)continue;const p=s[o],l=s[u];i.has(l.name)||i.set(l.name,[]),i.get(l.name).push(p)}return i}const jf=new Set(["Switch","Merge","Enter","Exit","NextIteration","StatelessIf","StatelessWhile","if","While"]),qf=new Set(["NonMaxSuppressionV2","NonMaxSuppressionV3","NonMaxSuppressionV5","Where"]),Kf=new Set(["HashTable","HashTableV2","LookupTableImport","LookupTableImportV2","LookupTableFind","LookupTableFindV2","LookupTableSize","LookupTableSizeV2"]);function we(s){return jf.has(s.op)}function Wf(s){return qf.has(s.op)}function Hf(s){return Kf.has(s.op)}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class bt{get weightIds(){return this.parent?this.parent.weightIds:this._weightIds}get functionExecutorMap(){return this.parent?this.parent.functionExecutorMap:this._functionExecutorMap}get weightMap(){return this.parent?this.parent.weightMap:this._weightMap}set weightMap(e){const t=Object.keys(e).map(n=>e[n].map(r=>r.id));this._weightIds=[].concat(...t),this._weightMap=e}set resourceManager(e){this._resourceManager=e}get inputs(){return this._inputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get outputs(){return this._outputs.map(e=>({name:e.name,shape:e.attrParams.shape?e.attrParams.shape.value:void 0,dtype:e.attrParams.dtype?e.attrParams.dtype.value:void 0}))}get inputNodes(){return this._inputs.map(e=>e.signatureKey||e.name)}get outputNodes(){return this._outputs.map(e=>{const t=e.signatureKey||e.name;return e.defaultOutput?`${t}:${e.defaultOutput}`:t})}get functions(){return Object.keys(this._functions).reduce((e,t)=>(e[t]=this._functions[t].signature,e),{})}constructor(e,t){this.graph=e,this.parent=t,this.compiledMap=new Map,this.parseNodeNameCache=new Map,this._weightMap={},this.SEPARATOR=",",this._functions={},this._functionExecutorMap={},this.keepIntermediateTensors=!1,this._outputs=e.outputs,this._inputs=e.inputs,this._initNodes=e.initNodes,this._signature=e.signature,this._functions=e.functions,e.functions!=null&&Object.keys(e.functions).forEach(n=>{this._functionExecutorMap[n]=new bt(e.functions[n],this)})}getCompilationKey(e,t){const n=e.map(a=>a.name).sort(),r=t.map(a=>a.name).sort();return n.join(this.SEPARATOR)+"--"+r.join(this.SEPARATOR)}compile(e,t){const n=Ls(e,t,this.weightMap,this._initNodes),{missingInputs:r,dynamicNode:a,syncInputs:i}=n;if(a!=null)throw new Error(`This execution contains the node '${a.name}', which has the dynamic op '${a.op}'. Please use model.executeAsync() instead. Alternatively, to avoid the dynamic ops, specify the inputs [${i}]`);if(r.length>0){const p=t.map(m=>m.name),l=Object.keys(e);throw new Error(`Cannot compute the outputs [${p}] from the provided inputs [${l}]. Missing the following inputs: [${r}]`)}const o=Lf(this.graph,n),u=Bf(o);return{orderedNodes:o,nodeLiveUntilMap:u}}cloneAndKeepTensor(e){if(e==null)return null;const t=e.clone();return me(t),t}cloneTensorList(e){return e?e.map(n=>this.cloneAndKeepTensor(n)):null}cloneTensorMap(e){return Object.fromEntries(Object.entries(e).map(([t,n])=>[t,this.cloneTensorList(n)]))}execute(e,t){this.disposeIntermediateTensors(),e=this.mapInputs(e);const n=Object.keys(e).sort();this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t);const r=n.map(d=>this.graph.nodes[ee(d)[0]]),a=t.map(d=>ee(d)[0]),i=new Set(a);let o=a.map(d=>this.graph.nodes[d]);o.length===0&&(o=this._outputs);const u=this.getCompilationKey(r,o);let p=this.compiledMap.get(u);p==null&&(p=this.compile(e,o),this.compiledMap.set(u,p));try{this.keepIntermediateTensors=Ue().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(d){this.keepIntermediateTensors=!1,console.warn(d.message)}const l={},m={};return ae(()=>{const d=new Vs(this.weightMap,l,m,this.functionExecutorMap,this.parseNodeNameCache),f=Object.assign({},this.weightMap);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap)),Object.keys(e).forEach(S=>{const[O,C]=ee(S,d),_=[];_[C]=e[S],f[O]=_,this.keepIntermediateTensors&&(this.clonedTensorsMap[O]=this.cloneTensorList(_))});const N=this.getFrozenTensorIds(f),{orderedNodes:$,nodeLiveUntilMap:T}=p;for(const S of $){if(f[S.name])continue;const O=Ps(S,f,d,this._resourceManager);if(lt(O))throw new Error(`The execution of the op '${S.op}' returned a promise. Please use model.executeAsync() instead.`);f[S.name]=O,this.keepIntermediateTensors&&(this.clonedTensorsMap[S.name]=this.cloneTensorList(O)),this.checkTensorForDisposalWithNodeLiveUntilInfo(S,f,d,N,i,T.get(S.name))}return this.parent==null&&d.dispose(N),t.map(S=>H(S,f,d))})}getFrozenTensorIds(e){const t=[].concat.apply([],Object.keys(e).map(n=>e[n]).map(n=>n.map(r=>r.id)));return new Set(t)}checkTensorForDisposal(e,t,n,r,a,i,o){if(!(we(t)||i.has(e))){for(const u of n[e])u!=null&&(o[u.id]=(o[u.id]||0)+t.children.length);for(const u of t.inputs){if(we(u))continue;const p=Is(u.name,n,r);if(p!=null)for(const l of p){if(!l||l.kept||a.has(l.id))continue;const m=o[l.id];m===1?(l.dispose(),delete o[l.id]):m!=null&&o[l.id]--}}}}checkTensorForDisposalWithNodeLiveUntilInfo(e,t,n,r,a,i){function o(u){return we(u)||a.has(u.name)}if(!(we(e)||i==null))for(const u of i){if(o(u))continue;const p=Is(u.name,t,n);for(const l of p)!l||l.kept||r.has(l.id)||l.dispose()}}async executeAsync(e,t){return this._executeAsync(e,t)}disposeIntermediateTensors(){this.clonedTensorsMap&&(Object.values(this.clonedTensorsMap).forEach(e=>{for(const t of e)t&&!t.isDisposed&&t.dispose()}),this.clonedTensorsMap=null)}getIntermediateTensors(){return this.clonedTensorsMap}async _executeAsync(e,t,n=!1,r={},a={}){this.disposeIntermediateTensors(),n||(e=this.mapInputs(e),this.checkInputs(e),this.checkInputShapeAndType(e),t=this.mapOutputs(t),this.checkOutputs(t));try{this.keepIntermediateTensors=Ue().getBool("KEEP_INTERMEDIATE_TENSORS")}catch(d){this.keepIntermediateTensors=!1,console.warn(d.message)}const i=new Vs(this.weightMap,r,a,this.functionExecutorMap,this.parseNodeNameCache);this.keepIntermediateTensors&&(this.clonedTensorsMap=this.cloneTensorMap(this.weightMap));const o=await this.executeWithControlFlow(e,i,t,n),u=t.map(d=>H(d,o,i)),p=u.map(d=>d.id),l=Object.keys(e).map(d=>e[d].id),m=new Set([...p,...l,...this.weightIds]);return Object.values(o).forEach(d=>{d.forEach(f=>{f&&!f.isDisposed&&!m.has(f.id)&&f.dispose()})}),this.parent==null&&i.dispose(m),u}async executeFunctionAsync(e,t,n){const r=e.reduce((a,i,o)=>(a[this.inputs[o].name]=i,a),{});return this._executeAsync(r,this.outputNodes,!0,t,n)}async executeWithControlFlow(e,t,n,r){const a=Object.keys(e),i=a.map(_=>this.graph.nodes[ee(_)[0]]),o=n.map(_=>ee(_)[0]),u=new Set(o);let p=o.map(_=>this.graph.nodes[_]);p.length===0&&(p=this._outputs);const{usedNodes:l,missingInputs:m,dynamicNode:d,syncInputs:f}=Ls(e,p,this.weightMap,this._initNodes),N=[...i,...this.graph.weights,...this._initNodes||[]].map(_=>({node:_,contexts:t.currentContext})),$=Object.assign({},this.weightMap);Object.keys(e).forEach(_=>{const[k,v]=ee(_),E=[];E[v]=e[_],$[k]=E});const T={},S=this.getFrozenTensorIds($),O={};for(;N.length>0;){const _=this.processStack(i,N,t,$,O,S,u,T,l);await Promise.all(_)}d==null&&!r&&console.warn("This model execution did not contain any nodes with control flow or dynamic output shapes. You can use model.execute() instead.");const C=p.filter(_=>!we(_)&&!H(_.name,$,t)).map(_=>_.name);if(C.length>0){let _="";throw d!=null&&(_=`Alternatively, to avoid the dynamic ops, use model.execute() and specify the inputs [${f}]`),new Error(`Cannot compute the outputs [${C}] from the provided inputs [${a}]. Consider providing the following inputs: [${m}]. ${_}`)}return $}processStack(e,t,n,r,a,i,o,u,p){const l=[];for(;t.length>0;){const m=t.pop();n.currentContext=m.contexts;let d="";if(m.node.op==="Enter"&&c("isConstant",m.node,r,n)&&([d]=pe(m.node.name,n)),r[m.node.name]==null){const f=Ps(m.node,r,n,this._resourceManager);d||([d]=pe(m.node.name,n));const N=n.currentContext;lt(f)?l.push(f.then($=>(r[d]=$,this.keepIntermediateTensors&&(this.clonedTensorsMap[d]=this.cloneTensorList($)),n.currentContext=N,this.checkTensorForDisposal(d,m.node,r,n,i,o,u),this.processChildNodes(m.node,t,n,r,a,p),$))):(r[d]=f,this.keepIntermediateTensors&&(this.clonedTensorsMap[d]=this.cloneTensorList(f)),this.checkTensorForDisposal(d,m.node,r,n,i,o,u),this.processChildNodes(m.node,t,n,r,a,p))}else this.processChildNodes(m.node,t,n,r,a,p)}return l}processChildNodes(e,t,n,r,a,i){e.children.forEach(o=>{const[u]=pe(o.name,n);a[u]||!i.has(o.name)||(o.op==="Merge"?o.inputNames.some(p=>!!H(p,r,n))&&(a[u]=!0,t.push({contexts:n.currentContext,node:o})):o.inputNames.every(p=>!!H(p,r,n))&&(a[u]=!0,t.push({contexts:n.currentContext,node:o})))})}dispose(){Object.keys(this.weightMap).forEach(e=>this.weightMap[e].forEach(t=>t.dispose()))}checkInputShapeAndType(e){Object.keys(e).forEach(t=>{const n=e[t],[r]=ee(t),a=this.graph.nodes[r];if(a.attrParams.shape&&a.attrParams.shape.value){const i=a.attrParams.shape.value,o=i.length===n.shape.length&&n.shape.every((u,p)=>i[p]===-1||i[p]===u);y(o,()=>`The shape of dict['${a.name}'] provided in model.execute(dict) must be [${i}], but was [${n.shape}]`)}a.attrParams.dtype&&a.attrParams.dtype.value&&y(n.dtype===a.attrParams.dtype.value,()=>`The dtype of dict['${a.name}'] provided in model.execute(dict) must be ${a.attrParams.dtype.value}, but was ${n.dtype}`)})}mapInputs(e){var t,n;const r={};for(const a in e){const i=(n=(t=this._signature)===null||t===void 0?void 0:t.inputs)===null||n===void 0?void 0:n[a];i!=null?r[i.name]=e[a]:r[a]=e[a]}return r}checkInputs(e){const t=Object.keys(e).filter(n=>{const[r]=ee(n);return this.graph.nodes[r]==null});if(t.length>0)throw new Error(`The dict provided in model.execute(dict) has keys: [${t}] that are not part of graph`)}mapOutputs(e){return e.map(t=>{var n,r;const a=(r=(n=this._signature)===null||n===void 0?void 0:n.outputs)===null||r===void 0?void 0:r[t];return a!=null?a.name:t},{})}checkOutputs(e){e.forEach(t=>{const[n]=ee(t);if(!this.graph.nodes[n])throw new Error(`The output '${t}' is not found in the graph`)})}}class Gf{constructor(e={},t={}){this.hashTableNameToHandle=e,this.hashTableMap=t}addHashTable(e,t){this.hashTableNameToHandle[e]=t.handle,this.hashTableMap[t.id]=t}getHashTableHandleByName(e){return this.hashTableNameToHandle[e]}getHashTableById(e){return this.hashTableMap[e]}dispose(){for(const e in this.hashTableMap)this.hashTableMap[e].clearAndClose(),delete this.hashTableMap[e];for(const e in this.hashTableNameToHandle)this.hashTableNameToHandle[e].dispose(),delete this.hashTableNameToHandle[e]}}/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Uf="?tfjs-format=file",Xf="model.json";class On{get modelVersion(){return this.version}get inputNodes(){return this.executor.inputNodes}get outputNodes(){return this.executor.outputNodes}get inputs(){return this.executor.inputs}get outputs(){return this.executor.outputs}get weights(){return this.executor.weightMap}get metadata(){return this.artifacts.userDefinedMetadata}get modelSignature(){return this.signature}get modelStructuredOutputKeys(){return this.structuredOutputKeys}constructor(e,t={},n=Sn){this.modelUrl=e,this.loadOptions=t,this.version="n/a",this.io=n,t==null&&(this.loadOptions={}),this.resourceManager=new Gf}findIOHandler(){const e=this.modelUrl;if(e.load!=null)this.handler=e;else if(this.loadOptions.requestInit!=null)this.handler=this.io.browserHTTPRequest(e,this.loadOptions);else{const t=this.io.getLoadHandlers(e,this.loadOptions);if(t.length===0)t.push(this.io.browserHTTPRequest(e,this.loadOptions));else if(t.length>1)throw new Error(`Found more than one (${t.length}) load handlers for URL '${[e]}'`);this.handler=t[0]}}load(){if(this.findIOHandler(),this.handler.load==null)throw new Error("Cannot proceed with model loading because the IOHandler provided does not have the `load` method implemented.");const e=this.handler.load();return lt(e)?e.then(t=>t.getWeightStream==null?this.loadSync(t):this.loadStreaming(t)):this.loadSync(e)}loadSync(e){const t=this.io.decodeWeights(e.weightData,e.weightSpecs);return this.loadWithWeightMap(e,t)}async loadStreaming(e){if(e.getWeightStream==null)throw new Error("Model artifacts missing streamWeights function");const t=await Ks(e.getWeightStream(),e.weightSpecs);return this.loadWithWeightMap(e,t)}loadWithWeightMap(e,t){this.artifacts=e;const n=this.artifacts.modelTopology;let r=this.artifacts.signature;if(this.artifacts.userDefinedMetadata!=null){const a=this.artifacts.userDefinedMetadata;a.signature!=null&&(r=a.signature),a.structuredOutputKeys!=null&&(this.structuredOutputKeys=a.structuredOutputKeys)}if(this.signature=r,this.version=`${n.versions.producer}.${n.versions.minConsumer}`,this.executor=new bt(Ds.Instance.transformGraph(n,this.signature)),this.executor.weightMap=this.convertTensorMapToTensorsMap(t),this.executor.resourceManager=this.resourceManager,e.modelInitializer!=null&&e.modelInitializer.node!=null){const a=Ds.Instance.transformGraph(e.modelInitializer);this.initializer=new bt(a),this.initializer.weightMap=this.executor.weightMap,this.initializer.resourceManager=this.resourceManager,this.initializerSignature=e.initializerSignature}return!0}async save(e,t){if(typeof e=="string"){const n=this.io.getSaveHandlers(e);if(n.length===0)throw new Error(`Cannot find any save handlers for URL '${e}'`);if(n.length>1)throw new Error(`Found more than one (${n.length}) save handlers for URL '${e}'`);e=n[0]}if(e.save==null)throw new Error("GraphModel.save() cannot proceed because the IOHandler provided does not have the `save` attribute defined.");return e.save(this.artifacts)}addStructuredOutputNames(e){if(this.structuredOutputKeys){const t=e instanceof _e?[e]:e,n={};return t.forEach((r,a)=>n[this.structuredOutputKeys[a]]=r),n}return e}predict(e,t){const n=this.execute(e,this.outputNodes);return this.addStructuredOutputNames(n)}async predictAsync(e,t){const n=await this.executeAsync(e,this.outputNodes);return this.addStructuredOutputNames(n)}normalizeInputs(e){var t;if(!(e instanceof _e)&&!Array.isArray(e)){const a=(t=this.signature)===null||t===void 0?void 0:t.inputs;if(a!=null)for(const i in a){const o=a[i];o.resourceId!=null&&(e[i]=this.resourceIdToCapturedInput[o.resourceId])}return e}e=Array.isArray(e)?e:[e];const n=Object.keys(this.resourceIdToCapturedInput).length;if(e.length+n!==this.inputNodes.length)throw new Error(`Input tensor count mismatch, the graph model has ${this.inputNodes.length-n} non-resource placeholders, while there are ${e.length} input tensors provided.`);let r=0;return this.inputNodes.reduce((a,i)=>{var o,u,p;const l=(p=(u=(o=this.signature)===null||o===void 0?void 0:o.inputs)===null||u===void 0?void 0:u[i])===null||p===void 0?void 0:p.resourceId;return l!=null?a[i]=this.resourceIdToCapturedInput[l]:a[i]=e[r++],a},{})}normalizeOutputs(e){return e=e||this.outputNodes,Array.isArray(e)?e:[e]}executeInitializerGraph(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.execute({},[]):this.initializer.execute({},Object.keys(this.initializerSignature.outputs))}async executeInitializerGraphAsync(){return this.initializer==null?[]:this.initializerSignature==null?this.initializer.executeAsync({},[]):this.initializer.executeAsync({},Object.keys(this.initializerSignature.outputs))}setResourceIdToCapturedInput(e){if(this.resourceIdToCapturedInput={},this.initializerSignature){const t=this.initializerSignature.outputs,n=Object.keys(t);for(let r=0;r<n.length;r++){const a=n[r],i=t[a];this.resourceIdToCapturedInput[i.resourceId]=e[r]}}}execute(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(this.executeInitializerGraph()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const n=this.executor.execute(e,t);return n.length>1?n:n[0]}async executeAsync(e,t){this.resourceIdToCapturedInput==null&&this.setResourceIdToCapturedInput(await this.executeInitializerGraphAsync()),e=this.normalizeInputs(e),t=this.normalizeOutputs(t);const n=await this.executor.executeAsync(e,t);return n.length>1?n:n[0]}getIntermediateTensors(){return this.executor.getIntermediateTensors()}disposeIntermediateTensors(){this.executor.disposeIntermediateTensors()}convertTensorMapToTensorsMap(e){return Object.keys(e).reduce((t,n)=>(t[n]=[e[n]],t),{})}dispose(){this.executor.dispose(),this.initializer&&(this.initializer.dispose(),this.resourceIdToCapturedInput&&qs(this.resourceIdToCapturedInput)),this.resourceManager.dispose()}}async function ty(s,e={},t=Sn){if(s==null)throw new Error("modelUrl in loadGraphModel() cannot be null. Please provide a url or an IOHandler that loads the model");e==null&&(e={}),e.fromTFHub&&typeof s=="string"&&(s=Yf(s));const n=new On(s,e,t);return await n.load(),n}function sy(s){if(s==null)throw new Error("modelUrl in loadGraphModelSync() cannot be null. Please provide model artifacts or an IOHandler that loads the model");let e;if(s instanceof Array){const[n,r]=s;if(!n)throw new Error("modelJSON must be the first element of the array");if(!r||!(r instanceof ArrayBuffer))throw new Error("An ArrayBuffer of weights must be the second element of the array");if(!("modelTopology"in n))throw new Error("Model JSON is missing 'modelTopology'");if(!("weightsManifest"in n))throw new Error("Model JSON is missing 'weightsManifest'");const a=Hs(n.weightsManifest),i=Ws(n,a,r);e=yt(i)}else if("load"in s)e=s;else if("modelTopology"in s&&"weightSpecs"in s&&"weightData"in s)e=yt(s);else throw new Error("Unknown model format");const t=new On(e);return t.load(),t}function Yf(s){return s.endsWith("/")||(s=s+"/"),`${s}${Xf}${Uf}`}export{ey as $,Fi as A,Y as B,Bi as C,qi as D,Wi as E,Gi as F,On as G,Yi as H,wm as I,Zs as J,_t as K,Sm as L,Ji as M,su as N,eu as O,ru as P,ou as Q,J as R,uu as S,ls as T,pu as U,mu as V,du as W,yu as X,ss as Y,vm as Z,Em as _,Ko as a,im as a$,Nu as a0,wu as a1,vu as a2,Eu as a3,gm as a4,Ae as a5,Nn as a6,Iu as a7,Js as a8,Au as a9,ty as aA,sy as aB,ac as aC,Xe as aD,rn as aE,lc as aF,hc as aG,on as aH,mt as aI,un as aJ,cn as aK,Nc as aL,id as aM,wc as aN,V as aO,Dm as aP,ze as aQ,pn as aR,_c as aS,kc as aT,ht as aU,xc as aV,zt as aW,dt as aX,Dc as aY,Cc as aZ,Pc as a_,Lu as aa,ke as ab,ye as ac,ju as ad,en as ae,us as af,tn as ag,$d as ah,Am as ai,sn as aj,dm as ak,kt as al,nn as am,ft as an,Et as ao,ad as ap,Nm as aq,Sn as ar,fn as as,Qu as at,Ju as au,ec as av,Pt as aw,ns as ax,od as ay,nc as az,Ho as b,de as b$,Lc as b0,Rc as b1,ue as b2,vt as b3,ln as b4,qc as b5,$e as b6,Wc as b7,Gc as b8,Je as b9,cs as bA,dn as bB,al as bC,cm as bD,rs as bE,as as bF,il as bG,cl as bH,ll as bI,hl as bJ,rd as bK,fl as bL,gl as bM,F as bN,Nl as bO,wl as bP,Sl as bQ,vl as bR,El as bS,an as bT,mn as bU,ud as bV,mm as bW,nd as bX,Ze as bY,yn as bZ,ps as b_,Yc as ba,Zc as bb,Mc as bc,tp as bd,op as be,up as bf,pp as bg,mp as bh,dp as bi,yp as bj,Fp as bk,hn as bl,jp as bm,is as bn,Wp as bo,Ye as bp,Qe as bq,Up as br,Mf as bs,uh as bt,ph as bu,Ee as bv,Qp as bw,Jp as bx,el as by,sl as bz,Uo as c,Ll as c0,cd as c1,Rl as c2,Ct as c3,re as c4,Ge as c5,gn as c6,Bl as c7,jl as c8,ql as c9,Wl as ca,He as cb,Td as cc,Gl as cd,Vt as ce,Xl as cf,Ql as cg,Jl as ch,Oe as ci,em as cj,tm as ck,ge as cl,bn as cm,Yo as d,Zo as e,Mo as f,ti as g,ni as h,ai as i,ii as j,ci as k,li as l,Xs as m,di as n,Ti as o,St as p,vi as q,Ei as r,Oi as s,Ys as t,Qs as u,Ai as v,rm as w,zi as x,Jf as y,Vi as z};
