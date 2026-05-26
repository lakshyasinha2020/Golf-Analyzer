import{a as Ro}from"./index-D5q4OXnS.js";function No(e,t){for(var n=0;n<t.length;n++){const s=t[n];if(typeof s!="string"&&!Array.isArray(s)){for(const r in s)if(r!=="default"&&!(r in e)){const o=Object.getOwnPropertyDescriptor(s,r);o&&Object.defineProperty(e,r,o.get?o:{enumerable:!0,get:()=>s[r]})}}}return Object.freeze(Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}))}/**
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
 */const Do=1e-7,_o=1e-4;class eh{constructor(t,n){this.backend=t,this.dataMover=n,this.data=new WeakMap,this.dataIdsCount=0}get(t){return this.data.has(t)||this.dataMover.moveData(this.backend,t),this.data.get(t)}set(t,n){this.dataIdsCount++,this.data.set(t,n)}has(t){return this.data.has(t)}delete(t){return this.dataIdsCount--,this.data.delete(t)}numDataIds(){return this.dataIdsCount}}class Bo{refCount(t){return nt("refCount")}incRef(t){return nt("incRef")}timerAvailable(){return!0}time(t){return nt("time")}read(t){return nt("read")}readSync(t){return nt("readSync")}readToGPU(t,n){return nt("readToGPU")}numDataIds(){return nt("numDataIds")}disposeData(t,n){return nt("disposeData")}write(t,n,s){return nt("write")}move(t,n,s,r,o){return nt("move")}createTensorFromGPUData(t,n,s){return nt("createTensorFromGPUData")}memory(){return nt("memory")}floatPrecision(){return nt("floatPrecision")}epsilon(){return this.floatPrecision()===32?Do:_o}dispose(){return nt("dispose")}}function nt(e){throw new Error(`'${e}' not yet implemented or not found in the registry. This kernel may not be supported by the tfjs backend you have chosen`)}/**
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
 */function is(e){let t=e.length,n=0;for(;t>0;)n=Math.random()*t|0,t--,yt(e,t,n)}function Oo(e,t){if(e.length!==t.length)throw new Error(`Array sizes must match to be shuffled together First array length was ${e.length}Second array length was ${t.length}`);let n=e.length,s=0;for(;n>0;)s=Math.random()*n|0,n--,yt(e,n,s),yt(t,n,s)}function ie(e,t,n){return Math.max(e,Math.min(t,n))}function Po(e){return e%2===0?e:e+1}function yt(e,t,n){const s=e[t];e[t]=e[n],e[n]=s}function Lo(e){let t=0;for(let n=0;n<e.length;n++)t+=e[n];return t}function Co(e,t){const n=Math.random();return t*n+(1-n)*e}function Uo(e,t){let n=0;for(let s=0;s<e.length;s++){const r=Number(e[s])-Number(t[s]);n+=r*r}return n}function T(e,t){if(!e)throw new Error(typeof t=="string"?t:t())}function fe(e,t,n=""){T($e(e,t),()=>n+` Shapes ${e} and ${t} must match`)}function Wo(e){T(e!=null,()=>"The input to the tensor constructor must be a non-null value.")}function F(e){if(e.length===0)return 1;let t=e[0];for(let n=1;n<e.length;n++)t*=e[n];return t}function Go(e){return e.length===0}function zo(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==null&&t[n]!==null&&e[n]!==t[n])return!1;return!0}function $e(e,t){if(e===t)return!0;if(e==null||t==null||e.length!==t.length)return!1;for(let n=0;n<e.length;n++)if(e[n]!==t[n])return!1;return!0}function Ee(e){return e%1===0}function qo(e){if(Math.tanh!=null)return Math.tanh(e);if(e===1/0)return 1;if(e===-1/0)return-1;{const t=Math.exp(2*e);return(t-1)/(t+1)}}function Vo(e){const t=Math.ceil(Math.sqrt(e));return[t,Math.ceil(e/t)]}function jo(e){const t=new Uint32Array(e);for(let n=0;n<e;++n)t[n]=n;return is(t),t}function oe(e,t){return t<=e.length?e:e+" ".repeat(t-e.length)}function Ho(e,t=r=>0,n,s){return new Promise((r,o)=>{let i=0;const a=()=>{if(e()){r();return}i++;const l=t(i);if(n!=null&&i>=n){o();return}s!=null?s(a,l):setTimeout(a,l)};a()})}function Ko(e,t){let n=1,s=-1;for(let o=0;o<e.length;++o)if(e[o]>=0)n*=e[o];else if(e[o]===-1){if(s!==-1)throw Error(`Shapes can only have 1 implicit size. Found -1 at dim ${s} and dim ${o}`);s=o}else if(e[o]<0)throw Error(`Shapes can not be < 0. Found ${e[o]} at dim ${o}`);if(s===-1){if(t>0&&t!==n)throw Error(`Size(${t}) must match the product of shape ${e}`);return e}if(n===0)throw Error(`Cannot infer the missing size in [${e}] when there are 0 elements`);if(t%n!==0)throw Error(`The implicit shape can't be a fractional number. Got ${t} / ${n}`);const r=e.slice();return r[s]=t/n,r}function Re(e,t){const n=t.length;return e=e==null?t.map((s,r)=>r):[].concat(e),T(e.every(s=>s>=-n&&s<n),()=>`All values in axis param must be in range [-${n}, ${n}) but got axis ${e}`),T(e.every(s=>Ee(s)),()=>`All values in axis param must be integers but got axis ${e}`),e.map(s=>s<0?n+s:s)}function Jo(e,t){const n=[],s=[],r=t!=null&&Array.isArray(t)&&t.length===0,o=t==null||r?null:Re(t,e).sort();let i=0;for(let a=0;a<e.length;++a){if(o!=null){if(o[i]===a&&e[a]!==1)throw new Error(`Can't squeeze axis ${a} since its dim '${e[a]}' is not 1`);(o[i]==null||o[i]>a)&&e[a]===1&&(n.push(e[a]),s.push(a)),o[i]<=a&&i++}e[a]!==1&&(n.push(e[a]),s.push(a))}return{newShape:n,keptDims:s}}function Tt(e,t){return j(e,t)}function j(e,t){let n=null;if(e==null||e==="float32")n=new Float32Array(t);else if(e==="int32")n=new Int32Array(t);else if(e==="bool")n=new Uint8Array(t);else if(e==="string")n=new Array(t);else throw new Error(`Unknown data type ${e}`);return n}function as(e,t){for(let n=0;n<e.length;n++){const s=e[n];if(isNaN(s)||!isFinite(s))throw Error(`A tensor of type ${t} being uploaded contains ${s}.`)}}function ls(e){return e==="bool"||e==="complex64"||e==="float32"||e==="int32"||e==="string"}function cs(e,t){return!(t==="complex64"||t==="float32"&&e!=="complex64"||t==="int32"&&e!=="float32"&&e!=="complex64"||t==="bool"&&e==="bool")}function ke(e){if(e==="float32"||e==="int32")return 4;if(e==="complex64")return 8;if(e==="bool")return 1;throw new Error(`Unknown dtype ${e}`)}function us(e){if(e==null)return 0;let t=0;return e.forEach(n=>t+=n.length),t}function Ne(e){return typeof e=="string"||e instanceof String}function hs(e){return typeof e=="boolean"}function fs(e){return typeof e=="number"}function de(e){return Array.isArray(e)?de(e[0]):e instanceof Float32Array?"float32":e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray?"int32":fs(e)?"float32":Ne(e)?"string":hs(e)?"bool":"float32"}function At(e){return!!(e&&e.constructor&&e.call&&e.apply)}function Te(e,t){for(let n=t;n<e;++n)if(e%n===0)return n;return e}function Z(e){const t=e.length;if(t<2)return[];const n=new Array(t-1);n[t-2]=e[t-1];for(let s=t-3;s>=0;--s)n[s]=n[s+1]*e[s+1];return n}function ds(e,t,n,s=!1){const r=new Array;if(t.length===1){const o=t[0]*(s?2:1);for(let i=0;i<o;i++)r[i]=n[e+i]}else{const o=t[0],i=t.slice(1),a=i.reduce((l,c)=>l*c)*(s?2:1);for(let l=0;l<o;l++)r[l]=ds(e+l*a,i,n,s)}return r}function zt(e,t,n=!1){if(e.length===0)return t[0];const s=e.reduce((r,o)=>r*o)*(n?2:1);if(s===0)return[];if(s!==t.length)throw new Error(`[${e}] does not match the input size ${t.length}${n?" for a complex tensor":""}.`);return ds(0,e,t,n)}function Xo(e,t){if(Array.isArray(e))return e;if(t==="float32")return e instanceof Float32Array?e:new Float32Array(e);if(t==="int32")return e instanceof Int32Array?e:new Int32Array(e);if(t==="bool"||t==="string")return Uint8Array.from(new Int32Array(e));throw new Error(`Unknown dtype ${t}`)}function gs(e,t){const n=pt(e,t);for(let s=0;s<n.length;s++)n[s]=1;return n}function pt(e,t){if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool")return new Uint8Array(e);throw new Error(`Unknown data type ${t}`)}function Zo(e,t){const n=e.reduce((s,r)=>s*r,1);if(t==null||t==="float32")return zt(e,new Float32Array(n));if(t==="int32")return zt(e,new Int32Array(n));if(t==="bool")return zt(e,new Uint8Array(n));throw new Error(`Unknown data type ${t}`)}function te(e){e.forEach(t=>{T(Number.isInteger(t)&&t>=0,()=>`Tensor must have a shape comprised of positive integers but got shape [${e}].`)})}function Ht(e,t,n){if(t===0)return 0;if(t===1)return e[0];let s=e[e.length-1];for(let r=0;r<e.length-1;++r)s+=n[r]*e[r];return s}function ge(e,t,n){if(t===0)return[];if(t===1)return[e];const s=new Array(t);for(let r=0;r<s.length-1;++r)s[r]=Math.floor(e/n[r]),e-=s[r]*n[r];return s[s.length-1]=e,s}function De(e){return e&&e.then&&typeof e.then=="function"}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const _n="tfjsflags";class Yo{constructor(t){this.global=t,this.flags={},this.flagRegistry={},this.urlFlags={},this.getQueryParams=Qo,this.populateURLFlags()}setPlatform(t,n){this.platform!=null&&(M().getBool("IS_TEST")||M().getBool("PROD")||console.warn(`Platform ${this.platformName} has already been set. Overwriting the platform with ${t}.`)),this.platformName=t,this.platform=n}registerFlag(t,n,s){if(this.flagRegistry[t]={evaluationFn:n,setHook:s},this.urlFlags[t]!=null){const r=this.urlFlags[t];M().getBool("IS_TEST")||M().getBool("PROD")||console.warn(`Setting feature override from URL ${t}: ${r}.`),this.set(t,r)}}async getAsync(t){return t in this.flags?this.flags[t]:(this.flags[t]=await this.evaluateFlag(t),this.flags[t])}get(t){if(t in this.flags)return this.flags[t];const n=this.evaluateFlag(t);if(De(n))throw new Error(`Flag ${t} cannot be synchronously evaluated. Please use getAsync() instead.`);return this.flags[t]=n,this.flags[t]}getNumber(t){return this.get(t)}getBool(t){return this.get(t)}getString(t){return this.get(t)}getFlags(){return this.flags}get features(){return this.flags}set(t,n){if(this.flagRegistry[t]==null)throw new Error(`Cannot set flag ${t} as it has not been registered.`);this.flags[t]=n,this.flagRegistry[t].setHook!=null&&this.flagRegistry[t].setHook(n)}evaluateFlag(t){if(this.flagRegistry[t]==null)throw new Error(`Cannot evaluate flag '${t}': no evaluation function found.`);return this.flagRegistry[t].evaluationFn()}setFlags(t){this.flags=Object.assign({},t)}reset(){this.flags={},this.urlFlags={},this.populateURLFlags()}populateURLFlags(){if(typeof this.global>"u"||typeof this.global.location>"u"||typeof this.global.location.search>"u")return;const t=this.getQueryParams(this.global.location.search);_n in t&&t[_n].split(",").forEach(s=>{const[r,o]=s.split(":");this.urlFlags[r]=ei(r,o)})}}function Qo(e){const t={};return e.replace(/[?&]([^=?&]+)(?:=([^&]*))?/g,(n,...s)=>(ti(t,s[0],s[1]),s.join("="))),t}function ti(e,t,n){e[decodeURIComponent(t)]=decodeURIComponent(n||"")}function ei(e,t){const n=t.toLowerCase();return n==="true"||n==="false"?n==="true":`${+n}`===n?+n:t}function M(){return ps}let ps=null;function ni(e){ps=e}/**
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
 */let Ce;function ms(){if(Ce==null){let e;if(typeof window<"u")e=window;else if(typeof global<"u")e=global;else if(typeof process<"u")e=process;else if(typeof self<"u")e=self;else throw new Error("Could not find a global object");Ce=e}return Ce}function si(){const e=ms();return e._tfGlobals==null&&(e._tfGlobals=new Map),e._tfGlobals}function hn(e,t){const n=si();if(n.has(e))return n.get(e);{const s=t();return n.set(e,s),n.get(e)}}const ws="Abs",nh="Acos",sh="Acosh",_e="Add",rh="AddN",oh="All",ih="Any",ah="ArgMax",lh="ArgMin",ch="Asin",uh="Asinh",hh="Atan",fh="Atanh",dh="Atan2",gh="AvgPool",ph="AvgPoolGrad",mh="AvgPool3D",wh="AvgPool3DGrad",yh="BatchMatMul",bh="BatchToSpaceND",Ih="Bincount",ys="BitwiseAnd",Sh="BroadcastTo",Eh="BroadcastArgs",fn="Cast",bs="Ceil",kh="ClipByValue",Is="Complex",ri="ComplexAbs",Th="Concat",Ah="Conv2D",xh="Conv2DBackpropFilter",vh="Conv2DBackpropInput",Mh="Conv3D",Fh="Conv3DBackpropFilterV2",$h="Conv3DBackpropInputV2",Rh="Cos",Nh="Cosh",Dh="Cumprod",_h="Cumsum",Bh="CropAndResize",Oh="DenseBincount",Ph="DepthToSpace",Lh="DepthwiseConv2dNative",Ch="DepthwiseConv2dNativeBackpropFilter",Uh="DepthwiseConv2dNativeBackpropInput",Wh="Diag",Gh="Dilation2D",zh="Dilation2DBackpropInput",qh="Dilation2DBackpropFilter",Vh="Draw",oi="RealDiv",jh="Einsum",ii="Elu",Hh="EluGrad",Kh="Erf",Ss="Equal",Es="Exp",Jh="ExpandDims",ks="Expm1",Xh="FFT",ai="Fill",Zh="FlipLeftRight",Ts="Floor",dn="FloorDiv",Yh="FusedBatchNorm",Qh="GatherV2",tf="GatherNd",As="Greater",xs="GreaterEqual",gn="Identity",ef="IFFT",nf="Imag",sf="IsFinite",rf="IsInf",of="IsNan",li="LeakyRelu",vs="Less",Ms="LessEqual",af="LinSpace",Fs="Log",lf="Log1p",cf="LogicalAnd",uf="LogicalNot",hf="LogicalOr",ff="LogicalXor",df="LogSoftmax",gf="LowerBound",pf="LRN",mf="LRNGrad",wf="MatrixBandPart",yf="Max",pn="Maximum",bf="MaxPool",If="MaxPoolGrad",Sf="MaxPool3D",Ef="MaxPool3DGrad",kf="MaxPoolWithArgmax",Tf="Mean",Af="Min",$s="Minimum",xf="MirrorPad",vf="Mod",Mf="Multinomial",mn="Multiply",ci="Neg",Rs="NotEqual",Ff="NonMaxSuppressionV3",$f="NonMaxSuppressionV4",Rf="NonMaxSuppressionV5",Nf="OnesLike",Df="OneHot",_f="Pack",Bf="PadV2",Of="Pool",ui="Pow",hi="Prelu",fi="Prod",Pf="RaggedGather",Lf="RaggedRange",Cf="RaggedTensorToTensor",Uf="Range",di="Real",Wf="Reciprocal",gi="Relu",pi="Reshape",Gf="ResizeNearestNeighbor",zf="ResizeNearestNeighborGrad",qf="ResizeBilinear",Vf="ResizeBilinearGrad",mi="Relu6",jf="Reverse",Hf="Round",Ns="Rsqrt",Kf="ScatterNd",Jf="TensorScatterUpdate",Xf="SearchSorted",Zf="Select",Yf="Selu",wi="Slice",Qf="Sin",td="Sinh",ed="Sign",wn="Sigmoid",nd="Softplus",yn="Sqrt",yi="Sum",sd="SpaceToBatchND",rd="SplitV",od="Softmax",id="SparseFillEmptyRows",ad="SparseReshape",ld="SparseSegmentMean",cd="SparseSegmentSum",ud="SparseToDense",Ds="SquaredDifference",hd="Square",_s="StaticRegexReplace",fd="StridedSlice",dd="StringNGrams",gd="StringSplit",pd="StringToHashBucketFast",bn="Sub",md="Tan",wd="Tanh",bi="Tile",yd="TopK",bd="Transform",Ii="Transpose",Id="Unique",Sd="Unpack",Ed="UnsortedSegmentSum",kd="UpperBound",Si="ZerosLike",Ei="Step",Td="FromPixels",Ad="RotateWithOffset",xd="_FusedMatMul",vd="FusedConv2D",Md="FusedDepthwiseConv2D";/**
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
 */function It(...e){M().getBool("IS_TEST")||M().getBool("PROD")||console.warn(...e)}function ki(...e){M().getBool("IS_TEST")||M().getBool("PROD")||console.log(...e)}/**
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
 */const Kt=hn("kernelRegistry",()=>new Map),ae=hn("gradRegistry",()=>new Map);function Bn(e,t){const n=In(e,t);return Kt.get(n)}function On(e){return ae.get(e)}function je(e){const t=Kt.entries(),n=[];for(;;){const{done:s,value:r}=t.next();if(s)break;const[o,i]=r,[a]=o.split("_");a===e&&n.push(i)}return n}function Ti(e){const{kernelName:t,backendName:n}=e,s=In(t,n);Kt.has(s)&&It(`The kernel '${t}' for backend '${n}' is already registered`),Kt.set(s,e)}function Fd(e){const{kernelName:t}=e;ae.has(t)&&M().getBool("DEBUG")&&It(`Overriding the gradient for '${t}'`),ae.set(t,e)}function $d(e,t){const n=In(e,t);if(!Kt.has(n))throw new Error(`The kernel '${e}' for backend '${t}' is not registered`);Kt.delete(n)}function Rd(e){if(!ae.has(e))throw new Error(`The gradient '${e}' for backend is not registered`);ae.delete(e)}function Nd(e,t){je(e).forEach(s=>{const r=Object.assign({},s,{backendName:t});Ti(r)})}function In(e,t){return`${t}_${e}`}/**
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
 */function Bs(e){return e instanceof Float32Array||e instanceof Int32Array||e instanceof Uint8Array||e instanceof Uint8ClampedArray}var Ue,Pn;function Ai(){if(Pn)return Ue;Pn=1,Ue=t;var e=null;try{e=new WebAssembly.Instance(new WebAssembly.Module(new Uint8Array([0,97,115,109,1,0,0,0,1,13,2,96,0,1,127,96,4,127,127,127,127,1,127,3,7,6,0,1,1,1,1,1,6,6,1,127,1,65,0,11,7,50,6,3,109,117,108,0,1,5,100,105,118,95,115,0,2,5,100,105,118,95,117,0,3,5,114,101,109,95,115,0,4,5,114,101,109,95,117,0,5,8,103,101,116,95,104,105,103,104,0,0,10,191,1,6,4,0,35,0,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,126,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,127,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,128,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,129,34,4,66,32,135,167,36,0,32,4,167,11,36,1,1,126,32,0,173,32,1,173,66,32,134,132,32,2,173,32,3,173,66,32,134,132,130,34,4,66,32,135,167,36,0,32,4,167,11])),{}).exports}catch{}function t(S,d,A){this.low=S|0,this.high=d|0,this.unsigned=!!A}t.prototype.__isLong__,Object.defineProperty(t.prototype,"__isLong__",{value:!0});function n(S){return(S&&S.__isLong__)===!0}t.isLong=n;var s={},r={};function o(S,d){var A,N,C;return d?(S>>>=0,(C=0<=S&&S<256)&&(N=r[S],N)?N:(A=a(S,(S|0)<0?-1:0,!0),C&&(r[S]=A),A)):(S|=0,(C=-128<=S&&S<128)&&(N=s[S],N)?N:(A=a(S,S<0?-1:0,!1),C&&(s[S]=A),A))}t.fromInt=o;function i(S,d){if(isNaN(S))return d?b:w;if(d){if(S<0)return b;if(S>=m)return B}else{if(S<=-y)return R;if(S+1>=y)return $}return S<0?i(-S,d).neg():a(S%g|0,S/g|0,d)}t.fromNumber=i;function a(S,d,A){return new t(S,d,A)}t.fromBits=a;var l=Math.pow;function c(S,d,A){if(S.length===0)throw Error("empty string");if(S==="NaN"||S==="Infinity"||S==="+Infinity"||S==="-Infinity")return w;if(typeof d=="number"?(A=d,d=!1):d=!!d,A=A||10,A<2||36<A)throw RangeError("radix");var N;if((N=S.indexOf("-"))>0)throw Error("interior hyphen");if(N===0)return c(S.substring(1),d,A).neg();for(var C=i(l(A,8)),O=w,q=0;q<S.length;q+=8){var et=Math.min(8,S.length-q),lt=parseInt(S.substring(q,q+et),A);if(et<8){var K=i(l(A,et));O=O.mul(K).add(i(lt))}else O=O.mul(C),O=O.add(i(lt))}return O.unsigned=d,O}t.fromString=c;function h(S,d){return typeof S=="number"?i(S,d):typeof S=="string"?c(S,d):a(S.low,S.high,typeof d=="boolean"?d:S.unsigned)}t.fromValue=h;var u=65536,f=1<<24,g=u*u,m=g*g,y=m/2,p=o(f),w=o(0);t.ZERO=w;var b=o(0,!0);t.UZERO=b;var I=o(1);t.ONE=I;var k=o(1,!0);t.UONE=k;var v=o(-1);t.NEG_ONE=v;var $=a(-1,2147483647,!1);t.MAX_VALUE=$;var B=a(-1,-1,!0);t.MAX_UNSIGNED_VALUE=B;var R=a(0,-2147483648,!1);t.MIN_VALUE=R;var E=t.prototype;return E.toInt=function(){return this.unsigned?this.low>>>0:this.low},E.toNumber=function(){return this.unsigned?(this.high>>>0)*g+(this.low>>>0):this.high*g+(this.low>>>0)},E.toString=function(d){if(d=d||10,d<2||36<d)throw RangeError("radix");if(this.isZero())return"0";if(this.isNegative())if(this.eq(R)){var A=i(d),N=this.div(A),C=N.mul(A).sub(this);return N.toString(d)+C.toInt().toString(d)}else return"-"+this.neg().toString(d);for(var O=i(l(d,6),this.unsigned),q=this,et="";;){var lt=q.div(O),K=q.sub(lt.mul(O)).toInt()>>>0,G=K.toString(d);if(q=lt,q.isZero())return G+et;for(;G.length<6;)G="0"+G;et=""+G+et}},E.getHighBits=function(){return this.high},E.getHighBitsUnsigned=function(){return this.high>>>0},E.getLowBits=function(){return this.low},E.getLowBitsUnsigned=function(){return this.low>>>0},E.getNumBitsAbs=function(){if(this.isNegative())return this.eq(R)?64:this.neg().getNumBitsAbs();for(var d=this.high!=0?this.high:this.low,A=31;A>0&&(d&1<<A)==0;A--);return this.high!=0?A+33:A+1},E.isZero=function(){return this.high===0&&this.low===0},E.eqz=E.isZero,E.isNegative=function(){return!this.unsigned&&this.high<0},E.isPositive=function(){return this.unsigned||this.high>=0},E.isOdd=function(){return(this.low&1)===1},E.isEven=function(){return(this.low&1)===0},E.equals=function(d){return n(d)||(d=h(d)),this.unsigned!==d.unsigned&&this.high>>>31===1&&d.high>>>31===1?!1:this.high===d.high&&this.low===d.low},E.eq=E.equals,E.notEquals=function(d){return!this.eq(d)},E.neq=E.notEquals,E.ne=E.notEquals,E.lessThan=function(d){return this.comp(d)<0},E.lt=E.lessThan,E.lessThanOrEqual=function(d){return this.comp(d)<=0},E.lte=E.lessThanOrEqual,E.le=E.lessThanOrEqual,E.greaterThan=function(d){return this.comp(d)>0},E.gt=E.greaterThan,E.greaterThanOrEqual=function(d){return this.comp(d)>=0},E.gte=E.greaterThanOrEqual,E.ge=E.greaterThanOrEqual,E.compare=function(d){if(n(d)||(d=h(d)),this.eq(d))return 0;var A=this.isNegative(),N=d.isNegative();return A&&!N?-1:!A&&N?1:this.unsigned?d.high>>>0>this.high>>>0||d.high===this.high&&d.low>>>0>this.low>>>0?-1:1:this.sub(d).isNegative()?-1:1},E.comp=E.compare,E.negate=function(){return!this.unsigned&&this.eq(R)?R:this.not().add(I)},E.neg=E.negate,E.add=function(d){n(d)||(d=h(d));var A=this.high>>>16,N=this.high&65535,C=this.low>>>16,O=this.low&65535,q=d.high>>>16,et=d.high&65535,lt=d.low>>>16,K=d.low&65535,G=0,ct=0,H=0,ot=0;return ot+=O+K,H+=ot>>>16,ot&=65535,H+=C+lt,ct+=H>>>16,H&=65535,ct+=N+et,G+=ct>>>16,ct&=65535,G+=A+q,G&=65535,a(H<<16|ot,G<<16|ct,this.unsigned)},E.subtract=function(d){return n(d)||(d=h(d)),this.add(d.neg())},E.sub=E.subtract,E.multiply=function(d){if(this.isZero())return w;if(n(d)||(d=h(d)),e){var A=e.mul(this.low,this.high,d.low,d.high);return a(A,e.get_high(),this.unsigned)}if(d.isZero())return w;if(this.eq(R))return d.isOdd()?R:w;if(d.eq(R))return this.isOdd()?R:w;if(this.isNegative())return d.isNegative()?this.neg().mul(d.neg()):this.neg().mul(d).neg();if(d.isNegative())return this.mul(d.neg()).neg();if(this.lt(p)&&d.lt(p))return i(this.toNumber()*d.toNumber(),this.unsigned);var N=this.high>>>16,C=this.high&65535,O=this.low>>>16,q=this.low&65535,et=d.high>>>16,lt=d.high&65535,K=d.low>>>16,G=d.low&65535,ct=0,H=0,ot=0,ye=0;return ye+=q*G,ot+=ye>>>16,ye&=65535,ot+=O*G,H+=ot>>>16,ot&=65535,ot+=q*K,H+=ot>>>16,ot&=65535,H+=C*G,ct+=H>>>16,H&=65535,H+=O*K,ct+=H>>>16,H&=65535,H+=q*lt,ct+=H>>>16,H&=65535,ct+=N*G+C*K+O*lt+q*et,ct&=65535,a(ot<<16|ye,ct<<16|H,this.unsigned)},E.mul=E.multiply,E.divide=function(d){if(n(d)||(d=h(d)),d.isZero())throw Error("division by zero");if(e){if(!this.unsigned&&this.high===-2147483648&&d.low===-1&&d.high===-1)return this;var A=(this.unsigned?e.div_u:e.div_s)(this.low,this.high,d.low,d.high);return a(A,e.get_high(),this.unsigned)}if(this.isZero())return this.unsigned?b:w;var N,C,O;if(this.unsigned){if(d.unsigned||(d=d.toUnsigned()),d.gt(this))return b;if(d.gt(this.shru(1)))return k;O=b}else{if(this.eq(R)){if(d.eq(I)||d.eq(v))return R;if(d.eq(R))return I;var q=this.shr(1);return N=q.div(d).shl(1),N.eq(w)?d.isNegative()?I:v:(C=this.sub(d.mul(N)),O=N.add(C.div(d)),O)}else if(d.eq(R))return this.unsigned?b:w;if(this.isNegative())return d.isNegative()?this.neg().div(d.neg()):this.neg().div(d).neg();if(d.isNegative())return this.div(d.neg()).neg();O=w}for(C=this;C.gte(d);){N=Math.max(1,Math.floor(C.toNumber()/d.toNumber()));for(var et=Math.ceil(Math.log(N)/Math.LN2),lt=et<=48?1:l(2,et-48),K=i(N),G=K.mul(d);G.isNegative()||G.gt(C);)N-=lt,K=i(N,this.unsigned),G=K.mul(d);K.isZero()&&(K=I),O=O.add(K),C=C.sub(G)}return O},E.div=E.divide,E.modulo=function(d){if(n(d)||(d=h(d)),e){var A=(this.unsigned?e.rem_u:e.rem_s)(this.low,this.high,d.low,d.high);return a(A,e.get_high(),this.unsigned)}return this.sub(this.div(d).mul(d))},E.mod=E.modulo,E.rem=E.modulo,E.not=function(){return a(~this.low,~this.high,this.unsigned)},E.and=function(d){return n(d)||(d=h(d)),a(this.low&d.low,this.high&d.high,this.unsigned)},E.or=function(d){return n(d)||(d=h(d)),a(this.low|d.low,this.high|d.high,this.unsigned)},E.xor=function(d){return n(d)||(d=h(d)),a(this.low^d.low,this.high^d.high,this.unsigned)},E.shiftLeft=function(d){return n(d)&&(d=d.toInt()),(d&=63)===0?this:d<32?a(this.low<<d,this.high<<d|this.low>>>32-d,this.unsigned):a(0,this.low<<d-32,this.unsigned)},E.shl=E.shiftLeft,E.shiftRight=function(d){return n(d)&&(d=d.toInt()),(d&=63)===0?this:d<32?a(this.low>>>d|this.high<<32-d,this.high>>d,this.unsigned):a(this.high>>d-32,this.high>=0?0:-1,this.unsigned)},E.shr=E.shiftRight,E.shiftRightUnsigned=function(d){if(n(d)&&(d=d.toInt()),d&=63,d===0)return this;var A=this.high;if(d<32){var N=this.low;return a(N>>>d|A<<32-d,A>>>d,this.unsigned)}else return d===32?a(A,0,this.unsigned):a(A>>>d-32,0,this.unsigned)},E.shru=E.shiftRightUnsigned,E.shr_u=E.shiftRightUnsigned,E.toSigned=function(){return this.unsigned?a(this.low,this.high,!1):this},E.toUnsigned=function(){return this.unsigned?this:a(this.low,this.high,!0)},E.toBytes=function(d){return d?this.toBytesLE():this.toBytesBE()},E.toBytesLE=function(){var d=this.high,A=this.low;return[A&255,A>>>8&255,A>>>16&255,A>>>24,d&255,d>>>8&255,d>>>16&255,d>>>24]},E.toBytesBE=function(){var d=this.high,A=this.low;return[d>>>24,d>>>16&255,d>>>8&255,d&255,A>>>24,A>>>16&255,A>>>8&255,A&255]},t.fromBytes=function(d,A,N){return N?t.fromBytesLE(d,A):t.fromBytesBE(d,A)},t.fromBytesLE=function(d,A){return new t(d[0]|d[1]<<8|d[2]<<16|d[3]<<24,d[4]|d[5]<<8|d[6]<<16|d[7]<<24,A)},t.fromBytesBE=function(d,A){return new t(d[4]<<24|d[5]<<16|d[6]<<8|d[7],d[0]<<24|d[1]<<16|d[2]<<8|d[3],A)},Ue}var Os=Ai();const Ps=Ro(Os),xi=No({__proto__:null,default:Ps},[Os]);/**
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
 */const $t=Ps||xi;function pe(e){return $t.fromString(e,!0,16)}const Ls=pe("c3a5c85c97cb3127"),Ft=pe("b492b66fbe98f273"),X=pe("9ae16a3b2f90404f");function He(e){return e.xor(e.shru(47))}function Cs(e,t,n){const s=e.slice(t,t+n);return $t.fromBytes(Array.from(s),!0,!0)}function L(e,t){return Cs(e,t,8)}function Ln(e,t){return Cs(e,t,4)}function V(e,t){return t===0?e:e.shru(t).or(e.shl(64-t))}function kt(e,t,n=pe("9ddfea08eb382d69")){let s=e.xor(t).mul(n);s=s.xor(s.shru(47));let r=t.xor(s).mul(n);return r=r.xor(r.shru(47)),r=r.mul(n),r}function vi(e,t,n,s,r,o){r=r.add(e),o=V(o.add(r).add(s),21);const i=r;return r=r.add(t),r=r.add(n),o=o.add(V(r,44)),[r.add(s),o.add(i)]}function be(e,t,n,s){return vi(L(e,t),L(e,t+8),L(e,t+16),L(e,t+24),n,s)}function Mi(e,t=e.length){if(t>=8){const n=X.add(t*2),s=L(e,0).add(X),r=L(e,t-8),o=V(r,37).mul(n).add(s),i=V(s,25).add(r).mul(n);return kt(o,i,n)}if(t>=4){const n=X.add(t*2),s=Ln(e,0);return kt(s.shl(3).add(t),Ln(e,t-4),n)}if(t>0){const n=e[0],s=e[t>>1],r=e[t-1],o=n+(s<<8),i=t+(r<<2);return He(X.mul(o).xor(Ls.mul(i))).mul(X)}return X}function Fi(e,t=e.length){const n=X.add(t*2),s=L(e,0).mul(Ft),r=L(e,8),o=L(e,t-8).mul(n),i=L(e,t-16).mul(X);return kt(V(s.add(r),43).add(V(o,30)).add(i),s.add(V(r.add(X),18)).add(o),n)}function $i(e,t=e.length){const n=X.add(t*2),s=L(e,0).mul(X),r=L(e,8),o=L(e,t-8).mul(n),i=L(e,t-16).mul(X),a=V(s.add(r),43).add(V(o,30)).add(i),l=kt(a,s.add(V(r.add(X),18)).add(o),n),c=L(e,16).mul(n),h=L(e,24),u=a.add(L(e,t-32)).mul(n),f=l.add(L(e,t-24)).mul(n);return kt(V(c.add(h),43).add(V(u,30)).add(f),c.add(V(h.add(s),18)).add(u),n)}function Us(e,t=e.length){const n=$t.fromNumber(81,!0);if(t<=32)return t<=16?Mi(e,t):Fi(e,t);if(t<=64)return $i(e,t);let s=n,r=n.mul(Ft).add(113),o=He(r.mul(X).add(113)).mul(X),i=[$t.UZERO,$t.UZERO],a=[$t.UZERO,$t.UZERO];s=s.mul(X).add(L(e,0));let l=0;const c=(t-1>>6)*64,h=c+(t-1&63)-63;do s=V(s.add(r).add(i[0]).add(L(e,l+8)),37).mul(Ft),r=V(r.add(i[1]).add(L(e,l+48)),42).mul(Ft),s=s.xor(a[1]),r=r.add(i[0]).add(L(e,l+40)),o=V(o.add(a[0]),33).mul(Ft),i=be(e,l,i[1].mul(Ft),s.add(a[0])),a=be(e,l+32,o.add(a[1]),r.add(L(e,l+16))),[o,s]=[s,o],l+=64;while(l!==c);const u=Ft.add(o.and(255).shl(1));return l=h,a[0]=a[0].add(t-1&63),i[0]=i[0].add(a[0]),a[0]=a[0].add(i[0]),s=V(s.add(r).add(i[0]).add(L(e,l+8)),37).mul(u),r=V(r.add(i[1]).add(L(e,l+48)),42).mul(u),s=s.xor(a[1].mul(9)),r=r.add(i[0].mul(9).add(L(e,l+40))),o=V(o.add(a[0]),33).mul(u),i=be(e,l,i[1].mul(u),s.add(a[0])),a=be(e,l+32,o.add(a[1]),r.add(L(e,l+16))),[o,s]=[s,o],kt(kt(i[0],a[0],u).add(He(r).mul(Ls)).add(o),kt(i[1],a[1],u).add(s),u)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ws(e,t){return t==="string"?Nt(e):me([e],t)}function Ri(e,t){return e instanceof Float32Array&&t==="float32"||e instanceof Int32Array&&t==="int32"||e instanceof Uint8Array&&t==="bool"}function me(e,t){if(t==="string")throw new Error("Cannot convert a string[] to a TypedArray");if(Array.isArray(e)&&(e=Jt(e)),M().getBool("DEBUG")&&as(e,t),Ri(e,t))return e;if(t==null||t==="float32"||t==="complex64")return new Float32Array(e);if(t==="int32")return new Int32Array(e);if(t==="bool"){const n=new Uint8Array(e.length);for(let s=0;s<n.length;++s)Math.round(e[s])!==0&&(n[s]=1);return n}else throw new Error(`Unknown data type ${t}`)}function le(){return M().platform.now()}function Ni(e,t){return M().platform.fetch(e,t)}function Nt(e,t="utf-8"){return t=t||"utf-8",M().platform.encode(e,t)}function Ae(e,t="utf-8"){return t=t||"utf-8",M().platform.decode(e,t)}function at(e){return M().platform.isTypedArray!=null?M().platform.isTypedArray(e):Bs(e)}function Jt(e,t=[],n=!1){if(t==null&&(t=[]),typeof e=="boolean"||typeof e=="number"||typeof e=="string"||De(e)||e==null||at(e)&&n)t.push(e);else if(Array.isArray(e)||at(e))for(let s=0;s<e.length;++s)Jt(e[s],t,n);else{let s=-1;for(const r of Object.keys(e))/^([1-9]+[0-9]*|0)$/.test(r)&&(s=Math.max(s,Number(r)));for(let r=0;r<=s;r++)Jt(e[r],t,n)}return t}const Dd=Object.freeze(Object.defineProperty({__proto__:null,arraysEqual:$e,arraysEqualWithNull:zo,assert:T,assertNonNegativeIntegerDimensions:te,assertNonNull:Wo,assertShapesMatch:fe,bytesFromStringArray:us,bytesPerElement:ke,checkConversionForErrors:as,clamp:ie,computeStrides:Z,convertBackendValuesAndArrayBuffer:Xo,createScalarValue:Ws,createShuffledIndices:jo,decodeString:Ae,distSquared:Uo,encodeString:Nt,fetch:Ni,fingerPrint64:Us,flatten:Jt,getArrayFromDType:j,getTypedArrayFromDType:Tt,hasEncodingLoss:cs,hexToLong:pe,indexToLoc:ge,inferDtype:de,inferFromImplicitShape:Ko,isBoolean:hs,isFunction:At,isInt:Ee,isNumber:fs,isPromise:De,isScalarShape:Go,isString:Ne,isTypedArray:at,isValidDtype:ls,locToIndex:Ht,makeOnesTypedArray:gs,makeZerosNestedTypedArray:Zo,makeZerosTypedArray:pt,nearestDivisor:Te,nearestLargerEven:Po,now:le,parseAxisParam:Re,randUniform:Co,repeatedTry:Ho,rightPad:oe,shuffle:is,shuffleCombo:Oo,sizeFromShape:F,sizeToSquarishShape:Vo,squeezeShape:Jo,sum:Lo,swap:yt,tanh:qo,toNestedArray:zt,toTypedArray:me},Symbol.toStringTag,{value:"Module"}));/**
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
 */class Di{constructor(t,n){this.backendTimer=t,this.logger=n,n==null&&(this.logger=new Bi)}profileKernel(t,n,s){let r;const o=()=>{r=s()};let i;const a=le();if(this.backendTimer.timerAvailable())i=this.backendTimer.time(o);else{o();for(const c of r)c.dataSync();i=Promise.resolve({kernelMs:le()-a})}if(M().getBool("CHECK_COMPUTATION_FOR_ERRORS"))for(let c=0;c<r.length;c++){const h=r[c];h.data().then(u=>{_i(u,h.dtype,t)})}return{kernelName:t,outputs:r,inputs:n,timeMs:i.then(c=>c.kernelMs),extraInfo:i.then(c=>c.getExtraProfileInfo!=null?c.getExtraProfileInfo():"")}}logKernelProfile(t){const{kernelName:n,outputs:s,timeMs:r,inputs:o,extraInfo:i}=t;s.forEach(a=>{Promise.all([a.data(),r,i]).then(l=>{this.logger.logKernelProfile(n,a,l[0],l[1],o,l[2])})})}}function _i(e,t,n){if(t!=="float32")return!1;for(let s=0;s<e.length;s++){const r=e[s];if(isNaN(r)||!isFinite(r))return console.warn(`Found ${r} in the result of '${n}'`),!0}return!1}class Bi{logKernelProfile(t,n,s,r,o,i){const a=typeof r=="number"?oe(`${r}ms`,9):r.error,l=oe(t,25),c=n.rank,h=n.size,u=oe(n.shape.toString(),14);let f="";for(const g in o){const m=o[g];if(m!=null){const y=m.shape||n.shape,p=y.length;f+=`${g}: ${p}D ${p>0?y:""} `}}console.log(`%c${l}	%c${a}	%c${c}D ${u}	%c${h}	%c${f}	%c${i}`,"font-weight:bold","color:red","color:blue","color: orange","color: green","color: steelblue")}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Oi(e,t,n){const s={},r={};for(let l=0;l<t.length;l++)s[t[l].id]=!0;for(let l=0;l<e.length;l++){const c=e[l],h=c.inputs;for(const u in h){const f=h[u];let g=!1;for(let m=0;m<t.length;m++)if(s[f.id]){c.outputs.forEach(y=>s[y.id]=!0),g=!0,r[c.id]=!0;break}if(g)break}}const o={};o[n.id]=!0;const i={};for(let l=e.length-1;l>=0;l--){const c=e[l],h=c.inputs;for(let u=0;u<c.outputs.length;u++)if(o[c.outputs[u].id]){for(const f in h)o[h[f].id]=!0,i[c.id]=!0;break}}const a=[];for(let l=0;l<e.length;l++){const c=e[l];if(r[c.id]&&i[c.id]){const h={};for(const f in c.inputs){const g=c.inputs[f];s[g.id]&&(h[f]=g)}const u=Object.assign({},c);u.inputs=h,u.outputs=c.outputs,a.push(u)}}return a}function Pi(e,t,n,s){for(let r=t.length-1;r>=0;r--){const o=t[r],i=[];if(o.outputs.forEach(l=>{const c=e[l.id];c!=null?i.push(c):i.push(null)}),o.gradient==null)throw new Error(`Cannot compute gradient: gradient function not found for ${o.kernelName}.`);const a=o.gradient(i);for(const l in o.inputs){if(!(l in a))throw new Error(`Cannot backprop through input ${l}. Available gradients found: ${Object.keys(a)}.`);const c=n(()=>a[l]());if(c.dtype!=="float32")throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input ${l} must have 'float32' dtype, but has '${c.dtype}'`);const h=o.inputs[l];if(!$e(c.shape,h.shape))throw new Error(`Error in gradient for op ${o.kernelName}. The gradient of input '${l}' has shape '${c.shape}', which does not match the shape of the input '${h.shape}'`);if(e[h.id]==null)e[h.id]=c;else{const u=e[h.id];e[h.id]=s(u,c),u.dispose()}}}}/**
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
 */const Cn=20,ee=3,We=7;function Li(e,t,n,s){const r=Z(t),o=Ci(e,t,n,r),i=t.length,a=Ie(e,t,n,r,o),l=["Tensor"];return s&&(l.push(`  dtype: ${n}`),l.push(`  rank: ${i}`),l.push(`  shape: [${t}]`),l.push("  values:")),l.push(a.map(c=>"    "+c).join(`
`)),l.join(`
`)}function Ci(e,t,n,s){const r=F(t),o=s[s.length-1],i=new Array(o).fill(0),a=t.length,l=n==="complex64"?se(e):e;if(a>1)for(let c=0;c<r/o;c++){const h=c*o;for(let u=0;u<o;u++)i[u]=Math.max(i[u],ne(l[h+u],0,n).length)}return i}function ne(e,t,n){let s;return Array.isArray(e)?s=`${parseFloat(e[0].toFixed(We))} + ${parseFloat(e[1].toFixed(We))}j`:Ne(e)?s=`'${e}'`:n==="bool"?s=Gs(e):s=parseFloat(e.toFixed(We)).toString(),oe(s,t)}function Gs(e){return e===0?"false":"true"}function Ie(e,t,n,s,r,o=!0){const i=n==="complex64"?2:1,a=t[0],l=t.length;if(l===0){if(n==="complex64"){const y=se(e);return[ne(y[0],0,n)]}return n==="bool"?[Gs(e[0])]:[e[0].toString()]}if(l===1){if(a>Cn){const p=ee*i;let w=Array.from(e.slice(0,p)),b=Array.from(e.slice((a-ee)*i,a*i));return n==="complex64"&&(w=se(w),b=se(b)),["["+w.map((I,k)=>ne(I,r[k],n)).join(", ")+", ..., "+b.map((I,k)=>ne(I,r[a-ee+k],n)).join(", ")+"]"]}return["["+(n==="complex64"?se(e):Array.from(e)).map((p,w)=>ne(p,r[w],n)).join(", ")+"]"]}const c=t.slice(1),h=s.slice(1),u=s[0]*i,f=[];if(a>Cn){for(let y=0;y<ee;y++){const p=y*u,w=p+u;f.push(...Ie(e.slice(p,w),c,n,h,r,!1))}f.push("...");for(let y=a-ee;y<a;y++){const p=y*u,w=p+u;f.push(...Ie(e.slice(p,w),c,n,h,r,y===a-1))}}else for(let y=0;y<a;y++){const p=y*u,w=p+u;f.push(...Ie(e.slice(p,w),c,n,h,r,y===a-1))}const g=l===2?",":"";f[0]="["+(a>0?f[0]+g:"");for(let y=1;y<f.length-1;y++)f[y]=" "+f[y]+g;let m=`,
`;for(let y=2;y<l;y++)m+=`
`;return f[f.length-1]=" "+f[f.length-1]+"]"+(o?"":m),f}function se(e){const t=[];for(let n=0;n<e.length;n+=2)t.push([e[n],e[n+1]]);return t}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */class xe{constructor(t,n,s){if(this.dtype=n,this.shape=t.slice(),this.size=F(t),s!=null){const r=s.length;T(r===this.size,()=>`Length of values '${r}' does not match the size inferred by the shape '${this.size}'.`)}if(n==="complex64")throw new Error("complex64 dtype TensorBuffers are not supported. Please create a TensorBuffer for the real and imaginary parts separately and call tf.complex(real, imag).");this.values=s||j(n,this.size),this.strides=Z(t)}set(t,...n){n.length===0&&(n=[0]),T(n.length===this.rank,()=>`The number of provided coordinates (${n.length}) must match the rank (${this.rank})`);const s=this.locToIndex(n);this.values[s]=t}get(...t){t.length===0&&(t=[0]);let n=0;for(const r of t){if(r<0||r>=this.shape[n]){const o=`Requested out of range element at ${t}.   Buffer shape=${this.shape}`;throw new Error(o)}n++}let s=t[t.length-1];for(let r=0;r<t.length-1;++r)s+=this.strides[r]*t[r];return this.values[s]}locToIndex(t){if(this.rank===0)return 0;if(this.rank===1)return t[0];let n=t[t.length-1];for(let s=0;s<t.length-1;++s)n+=this.strides[s]*t[s];return n}indexToLoc(t){if(this.rank===0)return[];if(this.rank===1)return[t];const n=new Array(this.shape.length);for(let s=0;s<n.length-1;++s)n[s]=Math.floor(t/this.strides[s]),t-=n[s]*this.strides[s];return n[n.length-1]=t,n}get rank(){return this.shape.length}toTensor(){return ht().makeTensor(this.values,this.shape,this.dtype)}}let ht=null,Wt=null;function Ui(e){ht=e}function Wi(e){Wt=e}class Y{constructor(t,n,s,r){this.kept=!1,this.isDisposedInternal=!1,this.shape=t.slice(),this.dtype=n||"float32",this.size=F(t),this.strides=Z(t),this.dataId=s,this.id=r,this.rankType=this.rank<5?this.rank.toString():"higher"}get rank(){return this.shape.length}async buffer(){const t=await this.data();return Wt.buffer(this.shape,this.dtype,t)}bufferSync(){return Wt.buffer(this.shape,this.dtype,this.dataSync())}async array(){const t=await this.data();return zt(this.shape,t,this.dtype==="complex64")}arraySync(){return zt(this.shape,this.dataSync(),this.dtype==="complex64")}async data(){this.throwIfDisposed();const t=ht().read(this.dataId);if(this.dtype==="string"){const n=await t;try{return n.map(s=>Ae(s))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}}return t}dataToGPU(t){return this.throwIfDisposed(),ht().readToGPU(this.dataId,t)}dataSync(){this.throwIfDisposed();const t=ht().readSync(this.dataId);if(this.dtype==="string")try{return t.map(n=>Ae(n))}catch{throw new Error("Failed to decode the string bytes into utf-8. To get the original bytes, call tensor.bytes().")}return t}async bytes(){this.throwIfDisposed();const t=await ht().read(this.dataId);return this.dtype==="string"?t:new Uint8Array(t.buffer)}dispose(){this.isDisposed||(this.kerasMask&&this.kerasMask.dispose(),ht().disposeTensor(this),this.isDisposedInternal=!0)}get isDisposed(){return this.isDisposedInternal}throwIfDisposed(){if(this.isDisposed)throw new Error("Tensor is disposed.")}print(t=!1){return Wt.print(this,t)}clone(){return this.throwIfDisposed(),Wt.clone(this)}toString(t=!1){const n=this.dataSync();return Li(n,this.shape,this.dtype,t)}cast(t){return this.throwIfDisposed(),Wt.cast(this,t)}variable(t=!0,n,s){return this.throwIfDisposed(),ht().makeVariable(this,t,n,s)}}Object.defineProperty(Y,Symbol.hasInstance,{value:e=>!!e&&e.data!=null&&e.dataSync!=null&&e.throwIfDisposed!=null});function zs(){return hn("Tensor",()=>Y)}zs();class ve extends Y{constructor(t,n,s,r){super(t.shape,t.dtype,t.dataId,r),this.trainable=n,this.name=s}assign(t){if(t.dtype!==this.dtype)throw new Error(`dtype of the new value (${t.dtype}) and previous value (${this.dtype}) must match`);if(!$e(t.shape,this.shape))throw new Error(`shape of the new value (${t.shape}) and previous value (${this.shape}) must match`);ht().disposeTensor(this),this.dataId=t.dataId,ht().incRef(this,null)}dispose(){ht().disposeVariable(this),this.isDisposedInternal=!0}}Object.defineProperty(ve,Symbol.hasInstance,{value:e=>e instanceof Y&&e.assign!=null&&e.assign instanceof Function});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */var Un;(function(e){e.R0="R0",e.R1="R1",e.R2="R2",e.R3="R3",e.R4="R4",e.R5="R5",e.R6="R6"})(Un||(Un={}));var Ke;(function(e){e.float32="float32",e.int32="int32",e.bool="int32",e.complex64="complex64"})(Ke||(Ke={}));var Je;(function(e){e.float32="float32",e.int32="int32",e.bool="bool",e.complex64="complex64"})(Je||(Je={}));var Xe;(function(e){e.float32="float32",e.int32="float32",e.bool="float32",e.complex64="complex64"})(Xe||(Xe={}));var Ze;(function(e){e.float32="complex64",e.int32="complex64",e.bool="complex64",e.complex64="complex64"})(Ze||(Ze={}));const Gi={float32:Xe,int32:Ke,bool:Je,complex64:Ze};function Be(e,t){if(e==="string"||t==="string"){if(e==="string"&&t==="string")return"string";throw new Error(`Can not upcast ${e} with ${t}`)}return Gi[e][t]}function _d(e){return Be(e,"int32")}function qs(e){return e!=null&&typeof e=="object"&&"texture"in e&&e.texture instanceof WebGLTexture}function Vs(e){return typeof GPUBuffer<"u"&&e!=null&&typeof e=="object"&&"buffer"in e&&e.buffer instanceof GPUBuffer}/**
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
 */function vt(e,t){if(e.dtype===t.dtype)return[e,t];const n=Be(e.dtype,t.dtype);return[e.cast(n),t.cast(n)]}function zi(e,t){T(e.dtype===t.dtype,()=>`The dtypes of the first(${e.dtype}) and second(${t.dtype}) input must match`)}function qi(e,t){return t.some(n=>n.id===e.id)}function Sn(e){const t=[];return js(e,t,new Set),t}function js(e,t,n){if(e==null)return;if(e instanceof Y){t.push(e);return}if(!Vi(e))return;const s=e;for(const r in s){const o=s[r];n.has(o)||(n.add(o),js(o,t,n))}}function Vi(e){return Array.isArray(e)||typeof e=="object"}const Bd=Object.freeze(Object.defineProperty({__proto__:null,assertTypesMatch:zi,getTensorsInContainer:Sn,isTensorInList:qi,makeTypesMatch:vt},Symbol.toStringTag,{value:"Module"}));/**
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
 */function Ge(e){return e.kernelName!=null}class Wn{constructor(){this.registeredVariables={},this.nextTapeNodeId=0,this.numBytes=0,this.numTensors=0,this.numStringTensors=0,this.numDataBuffers=0,this.gradientDepth=0,this.kernelDepth=0,this.scopeStack=[],this.numDataMovesStack=[],this.nextScopeId=0,this.tensorInfo=new WeakMap,this.profiling=!1,this.activeProfile={newBytes:0,newTensors:0,peakBytes:0,kernels:[],result:null,get kernelNames(){return Array.from(new Set(this.kernels.map(t=>t.name)))}}}dispose(){for(const t in this.registeredVariables)this.registeredVariables[t].dispose()}}class Xt{constructor(t){this.ENV=t,this.registry={},this.registryFactory={},this.pendingBackendInitId=0,this.state=new Wn}async ready(){if(this.pendingBackendInit!=null)return this.pendingBackendInit.then(()=>{});if(this.backendInstance!=null)return;const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n];if(await this.initializeBackend(s).success){await this.setBackend(s);return}}throw new Error("Could not initialize any backends, all backend initializations failed.")}get backend(){if(this.pendingBackendInit!=null)throw new Error(`Backend '${this.backendName}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);if(this.backendInstance==null){const{name:t,asyncInit:n}=this.initializeBackendsAndReturnBest();if(n)throw new Error(`The highest priority backend '${t}' has not yet been initialized. Make sure to await tf.ready() or await tf.setBackend() before calling other methods`);this.setBackend(t)}return this.backendInstance}backendNames(){return Object.keys(this.registryFactory)}findBackend(t){if(!(t in this.registry))if(t in this.registryFactory){const{asyncInit:n}=this.initializeBackend(t);if(n)return null}else return null;return this.registry[t]}findBackendFactory(t){return t in this.registryFactory?this.registryFactory[t].factory:null}registerBackend(t,n,s=1){return t in this.registryFactory?(It(`${t} backend was already registered. Reusing existing backend factory.`),!1):(this.registryFactory[t]={factory:n,priority:s},!0)}async setBackend(t){if(this.registryFactory[t]==null)throw new Error(`Backend name '${t}' not found in registry`);if(this.backendName=t,this.registry[t]==null){this.backendInstance=null;const{success:n,asyncInit:s}=this.initializeBackend(t);if(!(s?await n:n))return!1}return this.backendInstance=this.registry[t],this.setupRegisteredKernels(),this.profiler=new Di(this.backendInstance),!0}setupRegisteredKernels(){je(this.backendName).forEach(n=>{n.setupFunc!=null&&n.setupFunc(this.backendInstance)})}disposeRegisteredKernels(t){je(t).forEach(s=>{s.disposeFunc!=null&&s.disposeFunc(this.registry[t])})}initializeBackend(t){const n=this.registryFactory[t];if(n==null)throw new Error(`Cannot initialize backend ${t}, no registration found.`);try{const s=n.factory();if(s&&!(s instanceof Bo)&&typeof s.then=="function"){const r=++this.pendingBackendInitId,o=s.then(i=>r<this.pendingBackendInitId?!1:(this.registry[t]=i,this.pendingBackendInit=null,!0)).catch(i=>(r<this.pendingBackendInitId||(this.pendingBackendInit=null,It(`Initialization of backend ${t} failed`),It(i.stack||i.message)),!1));return this.pendingBackendInit=o,{success:o,asyncInit:!0}}else return this.registry[t]=s,{success:!0,asyncInit:!1}}catch(s){return It(`Initialization of backend ${t} failed`),It(s.stack||s.message),{success:!1,asyncInit:!1}}}removeBackend(t){if(!(t in this.registryFactory))throw new Error(`${t} backend not found in registry`);this.backendName===t&&this.pendingBackendInit!=null&&this.pendingBackendInitId++,t in this.registry&&(this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t]),delete this.registryFactory[t],this.backendName===t&&(this.pendingBackendInit=null,this.backendName=null,this.backendInstance=null)}getSortedBackends(){if(Object.keys(this.registryFactory).length===0)throw new Error("No backend found in registry.");return Object.keys(this.registryFactory).sort((t,n)=>this.registryFactory[n].priority-this.registryFactory[t].priority)}initializeBackendsAndReturnBest(){const t=this.getSortedBackends();for(let n=0;n<t.length;n++){const s=t[n],{success:r,asyncInit:o}=this.initializeBackend(s);if(o||r)return{name:s,asyncInit:o}}throw new Error("Could not initialize any backends, all backend initializations failed.")}moveData(t,n){const s=this.state.tensorInfo.get(n),r=s.backend,o=this.readSync(n),i=r.refCount(n);r.disposeData(n,!0),s.backend=t,t.move(n,o,s.shape,s.dtype,i),this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack[this.state.numDataMovesStack.length-1]++}tidy(t,n){let s=null;if(n==null){if(typeof t!="function")throw new Error("Please provide a function to tidy()");n=t}else{if(typeof t!="string"&&!(t instanceof String))throw new Error("When calling with two arguments, the first argument to tidy() must be a string");if(typeof n!="function")throw new Error("When calling with two arguments, the 2nd argument to tidy() must be a function");s=t}let r;return this.scopedRun(()=>this.startScope(s),()=>this.endScope(r),()=>(r=n(),r instanceof Promise&&console.error("Cannot return a Promise inside of tidy."),r))}scopedRun(t,n,s){t();try{const r=s();return n(),r}catch(r){throw n(),r}}nextTensorId(){return Xt.nextTensorId++}nextVariableId(){return Xt.nextVariableId++}clone(t){const n=x.runKernel(gn,{x:t}),s={x:t},r=i=>({x:()=>{const a="float32",l={x:i},c={dtype:a};return x.runKernel(fn,l,c)}}),o=[];return this.addTapeNode(this.state.activeScope.name,s,[n],r,o,{}),n}runKernel(t,n,s){if(this.backendName==null&&this.backend,!(Bn(t,this.backendName)!=null))throw new Error(`Kernel '${t}' not registered for backend '${this.backendName}'`);return this.runKernelFunc({kernelName:t,inputs:n,attrs:s})}shouldCheckForMemLeaks(){return this.ENV.getBool("IS_TEST")}checkKernelForMemLeak(t,n,s){const r=this.backend.numDataIds();let o=0;s.forEach(l=>{o+=l.dtype==="complex64"?3:1});const i=this.state.numDataMovesStack[this.state.numDataMovesStack.length-1],a=r-n-o-i;if(a>0)throw new Error(`Backend '${this.backendName}' has an internal memory leak (${a} data ids) after running '${t}'`)}runKernelFunc(t){let n,s=[];const r=this.isTapeOn(),o=this.state.numBytes,i=this.state.numTensors;this.shouldCheckForMemLeaks()&&this.state.numDataMovesStack.push(0);let a;this.backendName==null&&this.backend;let l;const c=Ge(t)?t.kernelName:this.state.activeScope!=null?this.state.activeScope.name:"";if(Ge(t)){const{kernelName:m,inputs:y,attrs:p}=t;this.backendName==null&&this.backend;const w=Bn(m,this.backendName);T(w!=null,()=>`Cannot find registered kernel '${m}' for backend '${this.backendName}'`),a=()=>{const b=this.backend.numDataIds();l=w.kernelFunc({inputs:y,attrs:p,backend:this.backend});const I=Array.isArray(l)?l:[l];this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(m,b,I);const k=I.map(v=>v.rank!=null?v:this.makeTensorFromTensorInfo(v));if(r){const v=this.getTensorsForGradient(m,y,k);s=this.saveTensorsForBackwardMode(v)}return k}}else{const{forwardFunc:m}=t,y=p=>{r&&(s=p.map(w=>this.keep(this.clone(w))))};a=()=>{const p=this.backend.numDataIds();l=this.tidy(()=>m(this.backend,y));const w=Array.isArray(l)?l:[l];return this.shouldCheckForMemLeaks()&&this.checkKernelForMemLeak(c,p,w),w}}const{inputs:h,attrs:u}=t,f=Ge(t)?null:t.backwardsFunc;let g;return this.scopedRun(()=>this.state.kernelDepth++,()=>this.state.kernelDepth--,()=>{!this.ENV.getBool("DEBUG")&&!this.state.profiling?n=a():(g=this.profiler.profileKernel(c,h,()=>a()),this.ENV.getBool("DEBUG")&&this.profiler.logKernelProfile(g),n=g.outputs)}),r&&this.addTapeNode(c,h,n,f,s,u),this.state.profiling&&this.state.activeProfile.kernels.push({name:c,bytesAdded:this.state.numBytes-o,totalBytesSnapshot:this.state.numBytes,tensorsAdded:this.state.numTensors-i,totalTensorsSnapshot:this.state.numTensors,inputShapes:Object.keys(h).map(m=>h[m]!=null?h[m].shape:null),outputShapes:n.map(m=>m.shape),kernelTimeMs:g.timeMs,extraInfo:g.extraInfo}),Array.isArray(l)?n:n[0]}saveTensorsForBackwardMode(t){return t.map(s=>this.keep(this.clone(s)))}getTensorsForGradient(t,n,s){const r=On(t);if(r!=null){const o=r.inputsToSave||[],i=r.outputsToSave||[];let a;r.saveAllInputs?(T(Array.isArray(n),()=>"saveAllInputs is true, expected inputs to be an array."),a=Object.keys(n).map(c=>n[c])):a=o.map(c=>n[c]);const l=s.filter((c,h)=>i[h]);return a.concat(l)}return[]}makeTensor(t,n,s,r){if(t==null)throw new Error("Values passed to engine.makeTensor() are null");s=s||"float32",r=r||this.backend;let o=t;s==="string"&&Ne(t[0])&&(o=t.map(l=>Nt(l)));const i=r.write(o,n,s),a=new Y(n,s,i,this.nextTensorId());if(this.trackTensor(a,r),s==="string"){const l=this.state.tensorInfo.get(i),c=us(o);this.state.numBytes+=c-l.bytes,l.bytes=c}return a}makeTensorFromDataId(t,n,s,r){s=s||"float32";const o={dataId:t,shape:n,dtype:s};return this.makeTensorFromTensorInfo(o,r)}makeTensorFromTensorInfo(t,n){const{dataId:s,shape:r,dtype:o}=t,i=new Y(r,o,s,this.nextTensorId());return this.trackTensor(i,n),i}makeVariable(t,n=!0,s,r){s=s||this.nextVariableId().toString(),r!=null&&r!==t.dtype&&(t=t.cast(r));const o=new ve(t,n,s,this.nextTensorId());if(this.state.registeredVariables[o.name]!=null)throw new Error(`Variable with name ${o.name} was already registered`);return this.state.registeredVariables[o.name]=o,this.incRef(o,this.backend),o}trackTensor(t,n){this.state.numTensors++,t.dtype==="string"&&this.state.numStringTensors++;let s=0;t.dtype!=="complex64"&&t.dtype!=="string"&&(s=t.size*ke(t.dtype)),this.state.numBytes+=s,this.state.tensorInfo.has(t.dataId)||(this.state.numDataBuffers++,this.state.tensorInfo.set(t.dataId,{backend:n||this.backend,dtype:t.dtype,shape:t.shape,bytes:s})),t instanceof ve||this.track(t)}incRef(t,n){this.trackTensor(t,n),this.backend.incRef(t.dataId)}removeDataId(t,n){this.state.tensorInfo.has(t)&&this.state.tensorInfo.get(t).backend===n&&(this.state.tensorInfo.delete(t),this.state.numDataBuffers--)}disposeTensor(t){if(!this.state.tensorInfo.has(t.dataId))return;const n=this.state.tensorInfo.get(t.dataId);if(this.state.numTensors--,t.dtype==="string"&&(this.state.numStringTensors--,this.state.numBytes-=n.bytes),t.dtype!=="complex64"&&t.dtype!=="string"){const s=t.size*ke(t.dtype);this.state.numBytes-=s}n.backend.disposeData(t.dataId)&&this.removeDataId(t.dataId,n.backend)}disposeVariables(){for(const t in this.state.registeredVariables){const n=this.state.registeredVariables[t];this.disposeVariable(n)}}disposeVariable(t){this.disposeTensor(t),this.state.registeredVariables[t.name]!=null&&delete this.state.registeredVariables[t.name]}memory(){const t=this.backend.memory();return t.numTensors=this.state.numTensors,t.numDataBuffers=this.state.numDataBuffers,t.numBytes=this.state.numBytes,this.state.numStringTensors>0&&(t.unreliable=!0,t.reasons==null&&(t.reasons=[]),t.reasons.push("Memory usage by string tensors is approximate (2 bytes per character)")),t}async profile(t){this.state.profiling=!0;const n=this.state.numBytes,s=this.state.numTensors;this.state.activeProfile.kernels=[],this.state.activeProfile.result=await t(),this.state.profiling=!1,this.state.activeProfile.peakBytes=Math.max(...this.state.activeProfile.kernels.map(r=>r.totalBytesSnapshot)),this.state.activeProfile.newBytes=this.state.numBytes-n,this.state.activeProfile.newTensors=this.state.numTensors-s;for(const r of this.state.activeProfile.kernels)r.kernelTimeMs=await r.kernelTimeMs,r.extraInfo=await r.extraInfo;return this.state.activeProfile}isTapeOn(){return this.state.gradientDepth>0&&this.state.kernelDepth===0}addTapeNode(t,n,s,r,o,i){const a={id:this.state.nextTapeNodeId++,kernelName:t,inputs:n,outputs:s,saved:o},l=On(t);l!=null&&(r=l.gradFunc),r!=null&&(a.gradient=c=>(c=c.map((h,u)=>{if(h==null){const f=s[u],g=pt(f.size,f.dtype);return this.makeTensor(g,f.shape,f.dtype)}return h}),r(c.length>1?c:c[0],o,i))),this.state.activeTape.push(a)}keep(t){return t.kept=!0,t}startTape(){this.state.gradientDepth===0&&(this.state.activeTape=[]),this.state.gradientDepth++}endTape(){this.state.gradientDepth--}startScope(t){const n={track:[],name:"unnamed scope",id:this.state.nextScopeId++};t&&(n.name=t),this.state.scopeStack.push(n),this.state.activeScope=n}endScope(t){const n=Sn(t),s=new Set(n.map(o=>o.id));for(let o=0;o<this.state.activeScope.track.length;o++){const i=this.state.activeScope.track[o];!i.kept&&!s.has(i.id)&&i.dispose()}const r=this.state.scopeStack.pop();this.state.activeScope=this.state.scopeStack.length===0?null:this.state.scopeStack[this.state.scopeStack.length-1],n.forEach(o=>{!o.kept&&o.scopeId===r.id&&this.track(o)})}gradients(t,n,s,r=!1){if(T(n.length>0,()=>"gradients() received an empty list of xs."),s!=null&&s.dtype!=="float32")throw new Error(`dy must have 'float32' dtype, but has '${s.dtype}'`);const o=this.scopedRun(()=>this.startTape(),()=>this.endTape(),()=>this.tidy("forward",t));T(o instanceof Y,()=>"The result y returned by f() must be a tensor.");const i=Oi(this.state.activeTape,n,o);if(!r&&i.length===0&&n.length>0)throw new Error("Cannot compute gradient of y=f(x) with respect to x. Make sure that the f you passed encloses all operations that lead from x to y.");return this.tidy("backward",()=>{const a={};a[o.id]=s??ji(o.shape),Pi(a,i,c=>this.tidy(c),Hi);const l=n.map(c=>a[c.id]);return this.state.gradientDepth===0&&(this.state.activeTape.forEach(c=>{for(const h of c.saved)h.dispose()}),this.state.activeTape=null),{value:o,grads:l}})}customGrad(t){return T(At(t),()=>"The f passed in customGrad(f) must be a function."),(...n)=>{T(n.every(a=>a instanceof Y),()=>"The args passed in customGrad(f)(x1, x2,...) must all be tensors");let s;const r={};n.forEach((a,l)=>{r[l]=a});const o=(a,l)=>(s=t(...n,l),T(s.value instanceof Y,()=>"The function f passed in customGrad(f) must return an object where `obj.value` is a tensor"),T(At(s.gradFunc),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function."),s.value),i=(a,l)=>{const c=s.gradFunc(a,l),h=Array.isArray(c)?c:[c];T(h.length===n.length,()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns the same number of tensors as inputs passed to f(...)."),T(h.every(f=>f instanceof Y),()=>"The function f passed in customGrad(f) must return an object where `obj.gradFunc` is a function that returns a list of only tensors.");const u={};return h.forEach((f,g)=>{u[g]=()=>f}),u};return this.runKernelFunc({forwardFunc:o,backwardsFunc:i,inputs:r})}}readSync(t){return this.state.tensorInfo.get(t).backend.readSync(t)}read(t){return this.state.tensorInfo.get(t).backend.read(t)}readToGPU(t,n){return this.state.tensorInfo.get(t).backend.readToGPU(t,n)}async time(t){const n=le(),s=await this.backend.time(t);return s.wallMs=le()-n,s}track(t){return this.state.activeScope!=null&&(t.scopeId=this.state.activeScope.id,this.state.activeScope.track.push(t)),t}get registeredVariables(){return this.state.registeredVariables}reset(){this.pendingBackendInitId++,this.state.dispose(),this.ENV.reset(),this.state=new Wn;for(const t in this.registry)this.disposeRegisteredKernels(t),this.registry[t].dispose(),delete this.registry[t];this.backendName=null,this.backendInstance=null,this.pendingBackendInit=null}}Xt.nextTensorId=0;Xt.nextVariableId=0;function ji(e){const t=gs(F(e),"float32");return x.makeTensor(t,e,"float32")}function Hs(){const e=ms();if(e._tfengine==null){const t=new Yo(e);e._tfengine=new Xt(t)}return ni(e._tfengine.ENV),Ui(()=>e._tfengine),e._tfengine}const x=Hs();function Hi(e,t){const n={a:e,b:t};return x.runKernel(_e,n)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Ki(){return typeof navigator<"u"&&navigator!=null}let Ye;function Ji(e){Ye=e}function Xi(e){if(Ye!==void 0)return Ye;if(e||Ki()){if(e||(e=navigator),e.product==="ReactNative")return!0;const t=e.userAgent||e.vendor||(typeof window<"u"?window.opera:"");if(!t){const n=e;return n.userAgentData&&n.userAgentData.mobile}return/(android|bb\d+|meego).+mobile|avantgo|bada\/|blackberry|blazer|compal|elaine|fennec|hiptop|iemobile|ip(hone|od)|iris|kindle|lge |maemo|midp|mmp|mobile.+firefox|netfront|opera m(ob|in)i|palm( os)?|phone|p(ixi|re)\/|plucker|pocket|psp|series(4|6)0|symbian|treo|up\.(browser|link)|vodafone|wap|windows ce|xda|xiino/i.test(t)||/1207|6310|6590|3gso|4thp|50[1-6]i|770s|802s|a wa|abac|ac(er|oo|s\-)|ai(ko|rn)|al(av|ca|co)|amoi|an(ex|ny|yw)|aptu|ar(ch|go)|as(te|us)|attw|au(di|\-m|r |s )|avan|be(ck|ll|nq)|bi(lb|rd)|bl(ac|az)|br(e|v)w|bumb|bw\-(n|u)|c55\/|capi|ccwa|cdm\-|cell|chtm|cldc|cmd\-|co(mp|nd)|craw|da(it|ll|ng)|dbte|dc\-s|devi|dica|dmob|do(c|p)o|ds(12|\-d)|el(49|ai)|em(l2|ul)|er(ic|k0)|esl8|ez([4-7]0|os|wa|ze)|fetc|fly(\-|_)|g1 u|g560|gene|gf\-5|g\-mo|go(\.w|od)|gr(ad|un)|haie|hcit|hd\-(m|p|t)|hei\-|hi(pt|ta)|hp( i|ip)|hs\-c|ht(c(\-| |_|a|g|p|s|t)|tp)|hu(aw|tc)|i\-(20|go|ma)|i230|iac( |\-|\/)|ibro|idea|ig01|ikom|im1k|inno|ipaq|iris|ja(t|v)a|jbro|jemu|jigs|kddi|keji|kgt( |\/)|klon|kpt |kwc\-|kyo(c|k)|le(no|xi)|lg( g|\/(k|l|u)|50|54|\-[a-w])|libw|lynx|m1\-w|m3ga|m50\/|ma(te|ui|xo)|mc(01|21|ca)|m\-cr|me(rc|ri)|mi(o8|oa|ts)|mmef|mo(01|02|bi|de|do|t(\-| |o|v)|zz)|mt(50|p1|v )|mwbp|mywa|n10[0-2]|n20[2-3]|n30(0|2)|n50(0|2|5)|n7(0(0|1)|10)|ne((c|m)\-|on|tf|wf|wg|wt)|nok(6|i)|nzph|o2im|op(ti|wv)|oran|owg1|p800|pan(a|d|t)|pdxg|pg(13|\-([1-8]|c))|phil|pire|pl(ay|uc)|pn\-2|po(ck|rt|se)|prox|psio|pt\-g|qa\-a|qc(07|12|21|32|60|\-[2-7]|i\-)|qtek|r380|r600|raks|rim9|ro(ve|zo)|s55\/|sa(ge|ma|mm|ms|ny|va)|sc(01|h\-|oo|p\-)|sdk\/|se(c(\-|0|1)|47|mc|nd|ri)|sgh\-|shar|sie(\-|m)|sk\-0|sl(45|id)|sm(al|ar|b3|it|t5)|so(ft|ny)|sp(01|h\-|v\-|v )|sy(01|mb)|t2(18|50)|t6(00|10|18)|ta(gt|lk)|tcl\-|tdg\-|tel(i|m)|tim\-|t\-mo|to(pl|sh)|ts(70|m\-|m3|m5)|tx\-9|up(\.b|g1|si)|utst|v400|v750|veri|vi(rg|te)|vk(40|5[0-3]|\-v)|vm40|voda|vulc|vx(52|53|60|61|70|80|81|83|85|98)|w3c(\-| )|webc|whit|wi(g |nc|nw)|wmlb|wonu|x700|yas\-|your|zeto|zte\-/i.test(t.substr(0,4))}return!1}function Ks(){return typeof window<"u"&&window.document!=null||typeof WorkerGlobalScope<"u"}const Od=Object.freeze(Object.defineProperty({__proto__:null,isBrowser:Ks,isMobile:Xi,mockIsMobile:Ji},Symbol.toStringTag,{value:"Module"}));/**
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
 */const Q=M();Q.registerFlag("DEBUG",()=>!1,e=>{e&&console.warn("Debugging mode is ON. The output of every math call will be downloaded to CPU and checked for NaNs. This significantly impacts performance.")});Q.registerFlag("IS_BROWSER",()=>Ks());Q.registerFlag("IS_NODE",()=>typeof process<"u"&&typeof process.versions<"u"&&typeof process.versions.node<"u");Q.registerFlag("IS_CHROME",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Chrome/.test(navigator.userAgent)&&/Google Inc/.test(navigator.vendor));Q.registerFlag("IS_SAFARI",()=>typeof navigator<"u"&&navigator!=null&&navigator.userAgent!=null&&/Safari/.test(navigator.userAgent)&&/Apple/.test(navigator.vendor));Q.registerFlag("PROD",()=>!1);Q.registerFlag("TENSORLIKE_CHECK_SHAPE_CONSISTENCY",()=>Q.getBool("DEBUG"));Q.registerFlag("DEPRECATION_WARNINGS_ENABLED",()=>!0);Q.registerFlag("IS_TEST",()=>!1);Q.registerFlag("CHECK_COMPUTATION_FOR_ERRORS",()=>Q.getBool("DEBUG"));Q.registerFlag("WRAP_TO_IMAGEBITMAP",()=>!1);Q.registerFlag("CANVAS2D_WILL_READ_FREQUENTLY_FOR_GPU",()=>!1);Q.registerFlag("USE_SETTIMEOUTCUSTOM",()=>!1);/**
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
 */function Js(e,t){let n=e;if(at(e))return t==="string"?[]:[e.length];if(qs(e)){const r=e.channels||"RGBA";return[e.height,e.width*r.length]}else if(Vs(e))return[e.buffer.size/(t==null?4:ke(t))];if(!Array.isArray(e))return[];const s=[];for(;Array.isArray(n)||at(n)&&t!=="string";)s.push(n.length),n=n[0];return Array.isArray(e)&&M().getBool("TENSORLIKE_CHECK_SHAPE_CONSISTENCY")&&Xs(e,s,[]),s}function Xs(e,t,n){if(n=n||[],!Array.isArray(e)&&!at(e)){T(t.length===0,()=>`Element arr[${n.join("][")}] is a primitive, but should be an array/TypedArray of ${t[0]} elements`);return}T(t.length>0,()=>`Element arr[${n.join("][")}] should be a primitive, but is an array of ${e.length} elements`),T(e.length===t[0],()=>`Element arr[${n.join("][")}] should have ${t[0]} elements, but has ${e.length} elements`);const s=t.slice(1);for(let r=0;r<e.length;++r)Xs(e[r],s,n.concat(r))}function Gn(e,t,n,s){if(e!=="string_or_numeric"){if(e==null)throw new Error("Expected dtype cannot be null.");if(e!=="numeric"&&e!==t||e==="numeric"&&t==="string")throw new Error(`Argument '${n}' passed to '${s}' must be ${e} tensor, but got ${t} tensor`)}}function D(e,t,n,s="numeric"){if(e instanceof zs())return Gn(s,e.dtype,t,n),e;let r=de(e);if(r!=="string"&&["bool","int32","float32"].indexOf(s)>=0&&(r=s),Gn(s,r,t,n),e==null||!at(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string"){const l=e==null?"null":e.constructor.name;throw new Error(`Argument '${t}' passed to '${n}' must be a Tensor or TensorLike, but got '${l}'`)}const o=Js(e,r);!at(e)&&!Array.isArray(e)&&(e=[e]);const a=r!=="string"?me(e,r):Jt(e,[],!0);return x.makeTensor(a,o,r)}function Zi(e,t,n,s="numeric"){if(!Array.isArray(e))throw new Error(`Argument ${t} passed to ${n} must be a \`Tensor[]\` or \`TensorLike[]\``);return e.map((o,i)=>D(o,`${t}[${i}]`,n,s))}/**
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
 */const Yi="__op";function W(e){const t=Object.keys(e);if(t.length!==1)throw new Error(`Please provide an object with a single key (operation name) mapping to a function. Got an object with ${t.length} keys.`);let n=t[0];const s=e[n];n.endsWith("_")&&(n=n.substring(0,n.length-1)),n=n+Yi;const r=(...o)=>{x.startScope(n);try{const i=s(...o);return De(i)&&console.error("Cannot return a Promise inside of tidy."),x.endScope(i),i}catch(i){throw x.endScope(null),i}};return Object.defineProperty(r,"name",{value:n,configurable:!0}),r}/**
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
 */function Qi(e,t){const n=D(e,"real","complex"),s=D(t,"imag","complex");fe(n.shape,s.shape,`real and imag shapes, ${n.shape} and ${s.shape}, must match in call to tf.complex().`);const r={real:n,imag:s};return x.runKernel(Is,r)}const Zs=W({complex_:Qi});/**
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
 */function Ys(e,t,n,s){if(s==null)s=de(e);else if(s==="complex64")throw new Error("Cannot construct a complex64 tensor directly. Please use tf.complex(real, imag).");if(Vs(e)||qs(e)){if(s!=="float32"&&s!=="int32")throw new Error(`Creating tensor from GPU data only supports 'float32'|'int32' dtype, while the dtype is ${s}.`);return x.backend.createTensorFromGPUData(e,t||n,s)}if(!at(e)&&!Array.isArray(e)&&typeof e!="number"&&typeof e!="boolean"&&typeof e!="string")throw new Error("values passed to tensor(values) must be a number/boolean/string or an array of numbers/booleans/strings, or a TypedArray");if(t!=null){te(t);const r=F(t),o=F(n);T(r===o,()=>`Based on the provided shape, [${t}], the tensor should have ${r} values but has ${o}`);for(let i=0;i<n.length;++i){const a=n[i],l=i===n.length-1?a!==F(t.slice(i)):!0;T(n[i]===t[i]||!l,()=>`Error creating a new Tensor. Inferred shape (${n}) does not match the provided shape (${t}). `)}}return!at(e)&&!Array.isArray(e)&&(e=[e]),t=t||n,e=s!=="string"?me(e,s):Jt(e,[],!0),x.makeTensor(e,t,s)}/**
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
 */function ze(e,t,n){const s=Js(e,n);return Ys(e,t,s,n)}/**
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
 */const _t={float32:4,float16:2,int32:4,uint16:2,uint8:1,bool:1,complex64:8};class mt{static join(t){return new mt(t).slice()}constructor(t){if(this.shards=[],this.previousShardIndex=0,t==null||(t instanceof Array||(t=[t]),t=t.map(s=>at(s)?s.buffer:s),t.length===0))return;this.bufferUniformSize=t[0].byteLength;let n=0;for(let s=0;s<t.length;s++){const r=t[s];s!==t.length-1&&r.byteLength!==this.bufferUniformSize&&(this.bufferUniformSize=void 0);const o=n+r.byteLength;this.shards.push({buffer:r,start:n,end:o}),n=o}this.shards.length===0&&(this.byteLength=0),this.byteLength=this.shards[this.shards.length-1].end}slice(t=0,n=this.byteLength){if(this.shards.length===0)return new ArrayBuffer(0);if(t=isNaN(Number(t))?0:t,n=isNaN(Number(n))?0:n,t=Math.max(0,t),n=Math.min(this.byteLength,n),n<=t)return new ArrayBuffer(0);const s=this.findShardForByte(t);if(s===-1)throw new Error(`Could not find start shard for byte ${t}`);const r=n-t,o=new ArrayBuffer(r),i=new Uint8Array(o);let a=0;for(let l=s;l<this.shards.length;l++){const c=this.shards[l],u=t+a-c.start,f=a,m=Math.min(n,c.end)-c.start,y=new Uint8Array(c.buffer,u,m-u);if(i.set(y,f),a+=y.length,n<c.end)break}return o}findShardForByte(t){if(this.shards.length===0||t<0||t>=this.byteLength)return-1;if(this.bufferUniformSize!=null)return this.previousShardIndex=Math.floor(t/this.bufferUniformSize),this.previousShardIndex;function n(r){return t<r.start?-1:t>=r.end?1:0}if(n(this.shards[this.previousShardIndex])===0)return this.previousShardIndex;const s=ta(this.shards,n);return s===-1?-1:(this.previousShardIndex=s,this.previousShardIndex)}}function ta(e,t){let n=0,s=e.length;for(;n<=s;){const r=Math.floor((s-n)/2)+n,o=t(e[r]);if(o===0)return r;o<0?s=r:n=r+1}return-1}/**
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
 */function Pd(){M().set("PROD",!0)}function Ld(){M().set("DEBUG",!0)}function Cd(){M().set("DEPRECATION_WARNINGS_ENABLED",!1),console.warn("TensorFlow.js deprecation warnings have been disabled.")}function Ud(e){M().getBool("DEPRECATION_WARNINGS_ENABLED")&&console.warn(e+" You can disable deprecation warnings with tf.disableDeprecationWarnings().")}function Wd(){x.disposeVariables()}function Gd(){return x}function zd(){return x.memory()}function qd(e){return x.profile(e)}function z(e,t){return x.tidy(e,t)}function it(e){Sn(e).forEach(n=>n.dispose())}function ea(e){return x.keep(e)}function Vd(e){return x.time(e)}function jd(e){return x.setBackend(e)}function Hd(){return x.ready()}function na(){return x.backendName}function Kd(e){x.removeBackend(e)}function Jd(e){return x.findBackend(e)}function Xd(e){return x.findBackendFactory(e)}function Zd(e,t,n=1){return x.registerBackend(e,t,n)}function sa(){return x.backend}function Yd(e,t){M().setPlatform(e,t)}/**
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
 */const xt=4;async function Qd(e,t){const n=[],s=[],r=Array.isArray(e)?e.map(i=>i.name):Object.keys(e);for(let i=0;i<r.length;++i){const a=r[i],l=Array.isArray(e)?e[i].tensor:e[a];if(l.dtype!=="float32"&&l.dtype!=="int32"&&l.dtype!=="bool"&&l.dtype!=="string"&&l.dtype!=="complex64")throw new Error(`Unsupported dtype in weight '${a}': ${l.dtype}`);const c={name:a,shape:l.shape,dtype:l.dtype};if(l.dtype==="string"){const h=new Promise(async u=>{const f=await l.bytes(),g=f.reduce((p,w)=>p+w.length,0)+xt*f.length,m=new Uint8Array(g);let y=0;for(let p=0;p<f.length;p++){const w=f[p],b=new Uint8Array(new Uint32Array([w.length]).buffer);m.set(b,y),y+=xt,m.set(w,y),y+=w.length}u(m)});s.push(h)}else s.push(l.data());t!=null&&(c.group=t),n.push(c)}const o=await Promise.all(s);return{data:aa(o),specs:n}}function ra(e,t){const n=new mt(e),s={};let r=0;for(const o of t){const i=oa(o,(a,l)=>n.slice(r+a,r+l));s[o.name]=Qs(o,n.slice(r,r+i)),r+=i}return s}function oa(e,t){const n=F(e.shape);let s;if("quantization"in e){const r=e.quantization;s=_t[r.dtype]}else if(e.dtype==="string"){let r=0;for(let o=0;o<n;o++)r+=xt+new Uint32Array(t(r,r+xt))[0];return r}else s=_t[e.dtype];return n*s}async function ia(e,t){const n=F(e.shape);let s;if("quantization"in e){const r=e.quantization;s=_t[r.dtype]}else if(e.dtype==="string"){let r=0;for(let o=0;o<n;o++)r+=xt+new Uint32Array(await t(r,r+xt))[0];return r}else s=_t[e.dtype];return n*s}function Qs(e,t){const n=e.name,s=e.dtype,r=e.shape,o=F(r);let i,a=0;if("quantization"in e){const l=e.quantization;if(l.dtype==="uint8"||l.dtype==="uint16"){if(!("min"in l&&"scale"in l))throw new Error(`Weight ${e.name} with quantization ${l.dtype} doesn't have corresponding metadata min and scale.`)}else if(l.dtype==="float16"){if(s!=="float32")throw new Error(`Weight ${e.name} is quantized with ${l.dtype} which only supports weights of type float32 not ${s}.`)}else throw new Error(`Weight ${e.name} has unknown quantization dtype ${l.dtype}. Supported quantization dtypes are: 'uint8', 'uint16', and 'float16'.`);const c=_t[l.dtype],h=l.dtype==="uint8"?new Uint8Array(t):new Uint16Array(t);if(s==="float32")if(l.dtype==="uint8"||l.dtype==="uint16"){i=new Float32Array(h.length);for(let u=0;u<h.length;u++){const f=h[u];i[u]=f*l.scale+l.min}}else if(l.dtype==="float16")i=ga()(h);else throw new Error(`Unsupported quantization type ${l.dtype} for weight type float32.`);else if(s==="int32"){if(l.dtype!=="uint8"&&l.dtype!=="uint16")throw new Error(`Unsupported quantization type ${l.dtype} for weight type int32.`);i=new Int32Array(h.length);for(let u=0;u<h.length;u++){const f=h[u];i[u]=Math.round(f*l.scale+l.min)}}else throw new Error(`Unsupported dtype in weight '${n}': ${s}`);a+=o*c}else if(s==="string"){const l=F(e.shape);i=[];for(let c=0;c<l;c++){const h=new Uint32Array(t.slice(a,a+xt))[0];a+=xt;const u=new Uint8Array(t.slice(a,a+h));i.push(u),a+=h}}else{const l=_t[s];if(s==="float32")i=new Float32Array(t);else if(s==="int32")i=new Int32Array(t);else if(s==="bool")i=new Uint8Array(t);else if(s==="complex64"){i=new Float32Array(t);const c=new Float32Array(i.length/2),h=new Float32Array(i.length/2);for(let m=0;m<c.length;m++)c[m]=i[m*2],h[m]=i[m*2+1];const u=ze(c,r,"float32"),f=ze(h,r,"float32"),g=Zs(u,f);return u.dispose(),f.dispose(),g}else throw new Error(`Unsupported dtype in weight '${n}': ${s}`);a+=o*l}return ze(i,r,s)}async function zn(e,t,n){let s=new Uint8Array(t);for(;s.byteLength<n;){const{done:r,value:o}=await e.read();if(r&&o==null){const a=n-s.byteLength;throw new Error(`Reader is done but ${a} bytes are still expected`)}const i=new Uint8Array(s.length+o.byteLength);i.set(s,0),i.set(new Uint8Array(o),s.length),s=i}return s.buffer}async function tg(e,t){const n={},s=e.getReader();let r=new ArrayBuffer(0);for(const o of t){const i=await ia(o,async(c,h)=>(r=await zn(s,r,h),r.slice(c,h)));r=await zn(s,r,i);const a=r.slice(0,i);r=r.slice(i);const l=Qs(o,a);if(n[o.name]=l,na()==="webgpu"){const c=sa();"uploadToGPU"in c&&F(l.shape)>=M().get("WEBGPU_CPU_HANDOFF_SIZE_THRESHOLD")&&c.uploadToGPU(l.dataId)}}return n}function aa(e){if(e===null)throw new Error(`Invalid input value: ${JSON.stringify(e)}`);let t=0;const n=[];e.forEach(o=>{if(t+=o.byteLength,n.push(o.byteLength===o.buffer.byteLength?o:new o.constructor(o)),!(o instanceof Float32Array||o instanceof Int32Array||o instanceof Uint8Array))throw new Error(`Unsupported TypedArray subtype: ${o.constructor.name}`)});const s=new Uint8Array(t);let r=0;return n.forEach(o=>{s.set(new Uint8Array(o.buffer),r),r+=o.byteLength}),s.buffer}const En=typeof Buffer<"u"&&(typeof Blob>"u"||typeof atob>"u"||typeof btoa>"u");function qn(e){return En?Buffer.byteLength(e,"utf8"):new Blob([e]).size}function la(e){if(En)return Buffer.from(e).toString("base64");const t=new Uint8Array(e);let n="";for(let s=0,r=t.length;s<r;s++)n+=String.fromCharCode(t[s]);return btoa(n)}function ca(e){if(En){const s=Buffer.from(e,"base64");return s.buffer.slice(s.byteOffset,s.byteOffset+s.byteLength)}const t=atob(e),n=new Uint8Array(t.length);for(let s=0;s<t.length;++s)n.set([t.charCodeAt(s)],s);return n.buffer}function eg(e){return mt.join(e)}function Vn(e){for(e=e.trim();e.endsWith("/");)e=e.slice(0,e.length-1);const n=e.split("/");return n[n.length-1]}function tr(e,t){const n={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy,weightsManifest:t};return e.signature!=null&&(n.signature=e.signature),e.userDefinedMetadata!=null&&(n.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(n.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(n.initializerSignature=e.initializerSignature),e.trainingConfig!=null&&(n.trainingConfig=e.trainingConfig),n}function ua(e,t,n){const s={modelTopology:e.modelTopology,format:e.format,generatedBy:e.generatedBy,convertedBy:e.convertedBy};if(e.trainingConfig!=null&&(s.trainingConfig=e.trainingConfig),e.weightsManifest!=null){if(!t)throw new Error("modelJSON has weightsManifest but weightSpecs is null");if(!n)throw new Error("modelJSON has weightsManifest but weightData is null");s.weightSpecs=t,s.weightData=n}return e.signature!=null&&(s.signature=e.signature),e.userDefinedMetadata!=null&&(s.userDefinedMetadata=e.userDefinedMetadata),e.modelInitializer!=null&&(s.modelInitializer=e.modelInitializer),e.initializerSignature!=null&&(s.initializerSignature=e.initializerSignature),s}async function er(e,t){let n,s;return e.weightsManifest!=null&&([n,s]=await t(e.weightsManifest)),ua(e,n,s)}function Oe(e){if(e.modelTopology instanceof ArrayBuffer)throw new Error("Expected JSON model topology, received ArrayBuffer.");return{dateSaved:new Date,modelTopologyType:"JSON",modelTopologyBytes:e.modelTopology==null?0:qn(JSON.stringify(e.modelTopology)),weightSpecsBytes:e.weightSpecs==null?0:qn(JSON.stringify(e.weightSpecs)),weightDataBytes:e.weightData==null?0:new mt(e.weightData).byteLength}}function jn(e){const t=[];for(const n of e)t.push(...n.weights);return t}function ha(){const e=n=>{let s=n<<13,r=0;for(;(s&8388608)===0;)r-=8388608,s<<=1;return s&=-8388609,r+=947912704,s|r},t=new Uint32Array(2048);t[0]=0;for(let n=1;n<1024;n++)t[n]=e(n);for(let n=1024;n<2048;n++)t[n]=939524096+(n-1024<<13);return t}function fa(){const e=new Uint32Array(64);e[0]=0,e[31]=1199570944,e[32]=2147483648,e[63]=3347054592;for(let t=1;t<31;t++)e[t]=t<<23;for(let t=33;t<63;t++)e[t]=2147483648+(t-32<<23);return e}function da(){const e=new Uint32Array(64);for(let t=0;t<64;t++)e[t]=1024;return e[0]=e[32]=0,e}function ga(){const e=ha(),t=fa(),n=da();return s=>{const r=new ArrayBuffer(4*s.length),o=new Uint32Array(r);for(let i=0;i<s.length;i++){const a=s[i],l=e[n[a>>10]+(a&1023)]+t[a>>10];o[i]=l}return new Float32Array(r)}}/**
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
 */class U{constructor(){this.saveRouters=[],this.loadRouters=[]}static getInstance(){return U.instance==null&&(U.instance=new U),U.instance}static registerSaveRouter(t){U.getInstance().saveRouters.push(t)}static registerLoadRouter(t){U.getInstance().loadRouters.push(t)}static getSaveHandlers(t){return U.getHandlers(t,"save")}static getLoadHandlers(t,n){return U.getHandlers(t,"load",n)}static getHandlers(t,n,s){const r=[];return(n==="load"?U.getInstance().loadRouters:U.getInstance().saveRouters).forEach(i=>{const a=i(t,s);a!==null&&r.push(a)}),r}}const ng=e=>U.registerSaveRouter(e),sg=e=>U.registerLoadRouter(e),rg=e=>U.getSaveHandlers(e),og=(e,t)=>U.getLoadHandlers(e,t);/**
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
 */const Qe="tensorflowjs",tn=1,Rt="models_store",Et="model_info_store";function nr(){if(!M().getBool("IS_BROWSER"))throw new Error("Failed to obtain IndexedDB factory because the current environmentis not a web browser.");const e=typeof window>"u"?self:window,t=e.indexedDB||e.mozIndexedDB||e.webkitIndexedDB||e.msIndexedDB||e.shimIndexedDB;if(t==null)throw new Error("The current browser does not appear to support IndexedDB.");return t}function en(e){const t=e.result;t.createObjectStore(Rt,{keyPath:"modelPath"}),t.createObjectStore(Et,{keyPath:"modelPath"})}class Bt{constructor(t){if(this.indexedDB=nr(),t==null||!t)throw new Error("For IndexedDB, modelPath must not be null, undefined or empty.");this.modelPath=t}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");return this.databaseAction(this.modelPath,t)}async load(){return this.databaseAction(this.modelPath)}databaseAction(t,n){return new Promise((s,r)=>{const o=this.indexedDB.open(Qe,tn);o.onupgradeneeded=()=>en(o),o.onsuccess=()=>{const i=o.result;if(n==null){const a=i.transaction(Rt,"readonly"),c=a.objectStore(Rt).get(this.modelPath);c.onsuccess=()=>{if(c.result==null)return i.close(),r(new Error(`Cannot find model with path '${this.modelPath}' in IndexedDB.`));s(c.result.modelArtifacts)},c.onerror=h=>(i.close(),r(c.error)),a.oncomplete=()=>i.close()}else{n.weightData=mt.join(n.weightData);const a=Oe(n),l=i.transaction(Et,"readwrite");let c=l.objectStore(Et),h;try{h=c.put({modelPath:this.modelPath,modelArtifactsInfo:a})}catch(f){return r(f)}let u;h.onsuccess=()=>{u=i.transaction(Rt,"readwrite");const f=u.objectStore(Rt);let g;try{g=f.put({modelPath:this.modelPath,modelArtifacts:n,modelArtifactsInfo:a})}catch(m){return r(m)}g.onsuccess=()=>s({modelArtifactsInfo:a}),g.onerror=m=>{c=l.objectStore(Et);const y=c.delete(this.modelPath);y.onsuccess=()=>(i.close(),r(g.error)),y.onerror=p=>(i.close(),r(g.error))}},h.onerror=f=>(i.close(),r(h.error)),l.oncomplete=()=>{u==null?i.close():u.oncomplete=()=>i.close()}}},o.onerror=i=>r(o.error)})}}Bt.URL_SCHEME="indexeddb://";const sr=e=>M().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Bt.URL_SCHEME)?pa(e.slice(Bt.URL_SCHEME.length)):null;U.registerSaveRouter(sr);U.registerLoadRouter(sr);function pa(e){return new Bt(e)}function ma(e){return e.startsWith(Bt.URL_SCHEME)?e.slice(Bt.URL_SCHEME.length):e}class wa{constructor(){this.indexedDB=nr()}async listModels(){return new Promise((t,n)=>{const s=this.indexedDB.open(Qe,tn);s.onupgradeneeded=()=>en(s),s.onsuccess=()=>{const r=s.result,o=r.transaction(Et,"readonly"),a=o.objectStore(Et).getAll();a.onsuccess=()=>{const l={};for(const c of a.result)l[c.modelPath]=c.modelArtifactsInfo;t(l)},a.onerror=l=>(r.close(),n(a.error)),o.oncomplete=()=>r.close()},s.onerror=r=>n(s.error)})}async removeModel(t){return t=ma(t),new Promise((n,s)=>{const r=this.indexedDB.open(Qe,tn);r.onupgradeneeded=()=>en(r),r.onsuccess=()=>{const o=r.result,i=o.transaction(Et,"readwrite"),a=i.objectStore(Et),l=a.get(t);let c;l.onsuccess=()=>{if(l.result==null)return o.close(),s(new Error(`Cannot find model with path '${t}' in IndexedDB.`));{const h=a.delete(t),u=()=>{c=o.transaction(Rt,"readwrite");const g=c.objectStore(Rt).delete(t);g.onsuccess=()=>n(l.result.modelArtifactsInfo),g.onerror=m=>s(l.error)};h.onsuccess=u,h.onerror=f=>(u(),o.close(),s(l.error))}},l.onerror=h=>(o.close(),s(l.error)),i.oncomplete=()=>{c==null?o.close():c.oncomplete=()=>o.close()}},r.onerror=o=>s(r.error)})}}/**
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
 */const bt="/",Gt="tensorflowjs_models",rr="info",ya="model_topology",ba="weight_specs",Ia="weight_data",Sa="model_metadata";function or(e){return{info:[Gt,e,rr].join(bt),topology:[Gt,e,ya].join(bt),weightSpecs:[Gt,e,ba].join(bt),weightData:[Gt,e,Ia].join(bt),modelMetadata:[Gt,e,Sa].join(bt)}}function ir(e){for(const t of Object.values(e))window.localStorage.removeItem(t)}function Ea(e){const t=e.split(bt);if(t.length<3)throw new Error(`Invalid key format: ${e}`);return t.slice(1,t.length-1).join(bt)}function ka(e){return e.startsWith(Ot.URL_SCHEME)?e.slice(Ot.URL_SCHEME.length):e}class Ot{constructor(t){if(!M().getBool("IS_BROWSER")||typeof window>"u"||typeof window.localStorage>"u")throw new Error("The current environment does not support local storage.");if(this.LS=window.localStorage,t==null||!t)throw new Error("For local storage, modelPath must not be null, undefined or empty.");this.modelPath=t,this.keys=or(this.modelPath)}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserLocalStorage.save() does not support saving model topology in binary formats yet.");{const n=JSON.stringify(t.modelTopology),s=JSON.stringify(t.weightSpecs),r=Oe(t),o=mt.join(t.weightData);try{this.LS.setItem(this.keys.info,JSON.stringify(r)),this.LS.setItem(this.keys.topology,n),this.LS.setItem(this.keys.weightSpecs,s),this.LS.setItem(this.keys.weightData,la(o));const i={format:t.format,generatedBy:t.generatedBy,convertedBy:t.convertedBy,signature:t.signature!=null?t.signature:void 0,userDefinedMetadata:t.userDefinedMetadata!=null?t.userDefinedMetadata:void 0,modelInitializer:t.modelInitializer!=null?t.modelInitializer:void 0,initializerSignature:t.initializerSignature!=null?t.initializerSignature:void 0,trainingConfig:t.trainingConfig!=null?t.trainingConfig:void 0};return this.LS.setItem(this.keys.modelMetadata,JSON.stringify(i)),{modelArtifactsInfo:r}}catch{throw ir(this.keys),new Error(`Failed to save model '${this.modelPath}' to local storage: size quota being exceeded is a possible cause of this failure: modelTopologyBytes=${r.modelTopologyBytes}, weightSpecsBytes=${r.weightSpecsBytes}, weightDataBytes=${r.weightDataBytes}.`)}}}async load(){const t=JSON.parse(this.LS.getItem(this.keys.info));if(t==null)throw new Error(`In local storage, there is no model with name '${this.modelPath}'`);if(t.modelTopologyType!=="JSON")throw new Error("BrowserLocalStorage does not support loading non-JSON model topology yet.");const n={},s=JSON.parse(this.LS.getItem(this.keys.topology));if(s==null)throw new Error(`In local storage, the topology of model '${this.modelPath}' is missing.`);n.modelTopology=s;const r=JSON.parse(this.LS.getItem(this.keys.weightSpecs));if(r==null)throw new Error(`In local storage, the weight specs of model '${this.modelPath}' are missing.`);n.weightSpecs=r;const o=this.LS.getItem(this.keys.modelMetadata);if(o!=null){const a=JSON.parse(o);n.format=a.format,n.generatedBy=a.generatedBy,n.convertedBy=a.convertedBy,a.signature!=null&&(n.signature=a.signature),a.userDefinedMetadata!=null&&(n.userDefinedMetadata=a.userDefinedMetadata),a.modelInitializer!=null&&(n.modelInitializer=a.modelInitializer),a.initializerSignature!=null&&(n.initializerSignature=a.initializerSignature),a.trainingConfig!=null&&(n.trainingConfig=a.trainingConfig)}const i=this.LS.getItem(this.keys.weightData);if(i==null)throw new Error(`In local storage, the binary weight values of model '${this.modelPath}' are missing.`);return n.weightData=ca(i),n}}Ot.URL_SCHEME="localstorage://";const ar=e=>M().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Ot.URL_SCHEME)?Ta(e.slice(Ot.URL_SCHEME.length)):null;U.registerSaveRouter(ar);U.registerLoadRouter(ar);function Ta(e){return new Ot(e)}class Aa{constructor(){T(M().getBool("IS_BROWSER"),()=>"Current environment is not a web browser"),T(typeof window>"u"||typeof window.localStorage<"u",()=>"Current browser does not appear to support localStorage"),this.LS=window.localStorage}async listModels(){const t={},n=Gt+bt,s=bt+rr;for(let r=0;r<this.LS.length;++r){const o=this.LS.key(r);if(o.startsWith(n)&&o.endsWith(s)){const i=Ea(o);t[i]=JSON.parse(this.LS.getItem(o))}}return t}async removeModel(t){t=ka(t);const n=or(t);if(this.LS.getItem(n.info)==null)throw new Error(`Cannot find model at path '${t}'`);const s=JSON.parse(this.LS.getItem(n.info));return ir(n),s}}/**
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
 */const qt="://";class J{constructor(){this.managers={}}static getInstance(){return J.instance==null&&(J.instance=new J),J.instance}static registerManager(t,n){T(t!=null,()=>"scheme must not be undefined or null."),t.endsWith(qt)&&(t=t.slice(0,t.indexOf(qt))),T(t.length>0,()=>"scheme must not be an empty string.");const s=J.getInstance();T(s.managers[t]==null,()=>`A model store manager is already registered for scheme '${t}'.`),s.managers[t]=n}static getManager(t){const n=J.getInstance().managers[t];if(n==null)throw new Error(`Cannot find model manager for scheme '${t}'`);return n}static getSchemes(){return Object.keys(J.getInstance().managers)}}function Se(e){if(e.indexOf(qt)===-1)throw new Error(`The url string provided does not contain a scheme. Supported schemes are: ${J.getSchemes().join(",")}`);return{scheme:e.split(qt)[0],path:e.split(qt)[1]}}async function lr(e,t,n=!1){T(e!==t,()=>`Old path and new path are the same: '${e}'`);const s=U.getLoadHandlers(e);T(s.length>0,()=>`Copying failed because no load handler is found for source URL ${e}.`),T(s.length<2,()=>`Copying failed because more than one (${s.length}) load handlers for source URL ${e}.`);const r=s[0],o=U.getSaveHandlers(t);T(o.length>0,()=>`Copying failed because no save handler is found for destination URL ${t}.`),T(o.length<2,()=>`Copying failed because more than one (${s.length}) save handlers for destination URL ${t}.`);const i=o[0],a=Se(e).scheme,l=Se(e).path,c=a===Se(e).scheme,h=await r.load();n&&c&&await J.getManager(a).removeModel(l);const u=await i.save(h);return n&&!c&&await J.getManager(a).removeModel(l),u.modelArtifactsInfo}async function ig(){const e=J.getSchemes(),t={};for(const n of e){const s=await J.getManager(n).listModels();for(const r in s){const o=n+qt+r;t[o]=s[r]}}return t}async function ag(e){const t=Se(e);return J.getManager(t.scheme).removeModel(t.path)}async function lg(e,t){return lr(e,t,!1)}async function cg(e,t){return lr(e,t,!0)}/**
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
 */class xa{constructor(){this.messageName="setTimeoutCustom",this.functionRefs=[],this.handledMessageCount=0,this.hasEventListener=!1}fetch(t,n){return fetch(t,n)}now(){return performance.now()}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Browser's encoder only supports utf-8, but got ${n}`);return this.textEncoder==null&&(this.textEncoder=new TextEncoder),this.textEncoder.encode(t)}decode(t,n){return new TextDecoder(n).decode(t)}setTimeoutCustom(t,n){if(typeof window>"u"||!M().getBool("USE_SETTIMEOUTCUSTOM")){setTimeout(t,n);return}this.functionRefs.push(t),setTimeout(()=>{window.postMessage({name:this.messageName,index:this.functionRefs.length-1},"*")},n),this.hasEventListener||(this.hasEventListener=!0,window.addEventListener("message",s=>{if(s.source===window&&s.data.name===this.messageName){s.stopPropagation();const r=this.functionRefs[s.data.index];r(),this.handledMessageCount++,this.handledMessageCount===this.functionRefs.length&&(this.functionRefs=[],this.handledMessageCount=0)}},!0))}isTypedArray(t){return Bs(t)}}if(M().get("IS_BROWSER")){M().setPlatform("browser",new xa);try{J.registerManager(Ot.URL_SCHEME,new Aa)}catch{}try{J.registerManager(Bt.URL_SCHEME,new wa)}catch{}}/**
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
 */const va={importFetch:()=>require("node-fetch")};let qe;class Ma{constructor(){this.util=require("util"),this.textEncoder=new this.util.TextEncoder}fetch(t,n){return M().global.fetch!=null?M().global.fetch(t,n):(qe==null&&(qe=va.importFetch()),qe(t,n))}now(){const t=process.hrtime();return t[0]*1e3+t[1]/1e6}encode(t,n){if(n!=="utf-8"&&n!=="utf8")throw new Error(`Node built-in encoder only supports utf-8, but got ${n}`);return this.textEncoder.encode(t)}decode(t,n){return t.length===0?"":new this.util.TextDecoder(n).decode(t)}isTypedArray(t){return this.util.types.isFloat32Array(t)||this.util.types.isInt32Array(t)||this.util.types.isUint8Array(t)||this.util.types.isUint8ClampedArray(t)}}M().get("IS_NODE")&&!M().get("IS_BROWSER")&&M().setPlatform("node",new Ma);/**
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
 */function st(e,t="float32",n){return t=t||"float32",te(e),new xe(e,t,n)}/**
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
 */function Fa(e,t){const n=D(e,"x","cast");if(!ls(t))throw new Error(`Failed to cast to unknown dtype ${t}`);if(t==="string"&&n.dtype!=="string"||t!=="string"&&n.dtype==="string")throw new Error("Only strings can be casted to strings");const s={x:n},r={dtype:t};return x.runKernel(fn,s,r)}const Me=W({cast_:Fa});/**
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
 */function $a(e){const n={x:D(e,"x","clone","string_or_numeric")};return x.runKernel(gn,n)}const cr=W({clone_:$a});/**
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
 */function Ra(e,t=!1){console.log(e.toString(t))}/**
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
 */Hs();const Na={buffer:st,cast:Me,clone:cr,print:Ra};Wi(Na);/**
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
 */function Da(e,t){let n=D(e,"a","add"),s=D(t,"b","add");[n,s]=vt(n,s);const r={a:n,b:s};return x.runKernel(_e,r)}const P=W({add_:Da});/**
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
 */function _a(e,t){let n=D(e,"a","floorDiv"),s=D(t,"b","floorDiv");[n,s]=vt(n,s);const r={a:n,b:s};return x.runKernel(dn,r)}const Ba=W({floorDiv_:_a});/**
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
 */function Oa(e,t){let n=D(e,"a","div"),s=D(t,"b","div");if([n,s]=vt(n,s),n.dtype==="int32"&&s.dtype==="int32")return Ba(n,s);const r={a:n,b:s},o={};return x.runKernel(oi,r,o)}const dt=W({div_:Oa});/**
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
 */function Pa(e,t){let n=D(e,"a","mul"),s=D(t,"b","mul");[n,s]=vt(n,s);const r={a:n,b:s};return x.runKernel(mn,r)}const _=W({mul_:Pa});/**
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
 */function La(e){const t=D(e,"x","abs");if(t.dtype==="complex64"){const n={x:t};return x.runKernel(ri,n)}else{const n={x:t};return x.runKernel(ws,n)}}const Ca=W({abs_:La});/**
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
 */function Ua(e,t,n,s,r="NHWC",o){const i=e[3],a=[...t,i],l=hr(r);return kn(e,a,n,o,s,null,null,l)}function Wa(e,t,n,s,r,o,i="channelsLast"){const[a,l]=ce(t);let c;if(i==="channelsLast")c=[a,l,e[3],e[3]];else if(i==="channelsFirst")c=[a,l,e[1],e[1]];else throw new Error(`Unknown dataFormat ${i}`);return kn(e,c,n,s,r,o,!1,i)}function Ga(e,t,n,s,r,o,i="NDHWC"){const[a,l,c]=nn(t);let h,u;if(i==="NDHWC")u="channelsLast",h=[a,l,c,e[4],e[4]];else if(i==="NCDHW")u="channelsFirst",h=[a,l,c,e[1],e[1]];else throw new Error(`Unknown dataFormat ${i}`);return ur(e,h,n,s,r,!1,u,o)}function kn(e,t,n,s,r,o,i=!1,a="channelsLast"){let[l,c,h,u]=[-1,-1,-1,-1];if(a==="channelsLast")[l,c,h,u]=e;else if(a==="channelsFirst")[l,u,c,h]=e;else throw new Error(`Unknown dataFormat ${a}`);const[f,g,,m]=t,[y,p]=ce(n),[w,b]=ce(s),I=Vt(f,w),k=Vt(g,b),{padInfo:v,outHeight:$,outWidth:B}=Va(r,c,h,y,p,I,k,o,a),R=i?m*u:m;let E;return a==="channelsFirst"?E=[l,R,$,B]:a==="channelsLast"&&(E=[l,$,B,R]),{batchSize:l,dataFormat:a,inHeight:c,inWidth:h,inChannels:u,outHeight:$,outWidth:B,outChannels:R,padInfo:v,strideHeight:y,strideWidth:p,filterHeight:f,filterWidth:g,effectiveFilterHeight:I,effectiveFilterWidth:k,dilationHeight:w,dilationWidth:b,inShape:e,outShape:E,filterShape:t}}function ur(e,t,n,s,r,o=!1,i="channelsLast",a){let[l,c,h,u,f]=[-1,-1,-1,-1,-1];if(i==="channelsLast")[l,c,h,u,f]=e;else if(i==="channelsFirst")[l,f,c,h,u]=e;else throw new Error(`Unknown dataFormat ${i}`);const[g,m,y,,p]=t,[w,b,I]=nn(n),[k,v,$]=nn(s),B=Vt(g,k),R=Vt(m,v),E=Vt(y,$),{padInfo:S,outDepth:d,outHeight:A,outWidth:N}=ja(r,c,h,u,w,b,I,B,R,E,a),C=o?p*f:p;let O;return i==="channelsFirst"?O=[l,C,d,A,N]:i==="channelsLast"&&(O=[l,d,A,N,C]),{batchSize:l,dataFormat:i,inDepth:c,inHeight:h,inWidth:u,inChannels:f,outDepth:d,outHeight:A,outWidth:N,outChannels:C,padInfo:S,strideDepth:w,strideHeight:b,strideWidth:I,filterDepth:g,filterHeight:m,filterWidth:y,effectiveFilterDepth:B,effectiveFilterHeight:R,effectiveFilterWidth:E,dilationDepth:k,dilationHeight:v,dilationWidth:$,inShape:e,outShape:O,filterShape:t}}function za(e,t,n,s,r){s==null&&(s=Tn(e,t,n));const o=e[0],i=e[1],a=ue((o-t+2*s)/n+1,r),l=ue((i-t+2*s)/n+1,r);return[a,l]}function qa(e,t,n,s,r,o){r==null&&(r=Tn(e,t[0],s[0]));const i=[0,0,0,n];for(let a=0;a<3;a++)e[a]+2*r>=t[a]&&(i[a]=ue((e[a]-t[a]+2*r)/s[a]+1,o));return i}function Tn(e,t,n,s=1){const r=Vt(t,s);return Math.floor((e[0]*(n-1)-n+r)/2)}function ce(e){return typeof e=="number"?[e,e,e]:e.length===2?[e[0],e[1],1]:e}function nn(e){return typeof e=="number"?[e,e,e]:e}function Vt(e,t){return t<=1?e:e+(e-1)*(t-1)}function Va(e,t,n,s,r,o,i,a,l){let c,h,u;if(typeof e=="number"){c={top:e,bottom:e,left:e,right:e,type:e===0?"VALID":"NUMBER"};const g=za([t,n],o,s,e,a);h=g[0],u=g[1]}else if(e==="same"){h=Math.ceil(t/s),u=Math.ceil(n/r);const f=Math.max(0,(h-1)*s+o-t),g=Math.max(0,(u-1)*r+i-n),m=Math.floor(f/2),y=f-m,p=Math.floor(g/2),w=g-p;c={top:m,bottom:y,left:p,right:w,type:"SAME"}}else if(e==="valid")c={top:0,bottom:0,left:0,right:0,type:"VALID"},h=Math.ceil((t-o+1)/s),u=Math.ceil((n-i+1)/r);else if(typeof e=="object"){const f=l==="channelsLast"?e[1][0]:e[2][0],g=l==="channelsLast"?e[1][1]:e[2][1],m=l==="channelsLast"?e[2][0]:e[3][0],y=l==="channelsLast"?e[2][1]:e[3][1];c={top:f,bottom:g,left:m,right:y,type:f===0&&g===0&&m===0&&y===0?"VALID":"EXPLICIT"},h=ue((t-o+f+g)/s+1,a),u=ue((n-i+m+y)/r+1,a)}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:c,outHeight:h,outWidth:u}}function ja(e,t,n,s,r,o,i,a,l,c,h){let u,f,g,m;if(e==="valid"&&(e=0),typeof e=="number"){u={top:e,bottom:e,left:e,right:e,front:e,back:e,type:e===0?"VALID":"NUMBER"};const p=qa([t,n,s,1],[a,l,c],1,[r,o,i],e,h);f=p[0],g=p[1],m=p[2]}else if(e==="same"){f=Math.ceil(t/r),g=Math.ceil(n/o),m=Math.ceil(s/i);const y=(f-1)*r+a-t,p=(g-1)*o+l-n,w=(m-1)*i+c-s,b=Math.floor(y/2),I=y-b,k=Math.floor(p/2),v=p-k,$=Math.floor(w/2),B=w-$;u={top:k,bottom:v,left:$,right:B,front:b,back:I,type:"SAME"}}else throw Error(`Unknown padding parameter: ${e}`);return{padInfo:u,outDepth:f,outHeight:g,outWidth:m}}function ue(e,t){if(!t)return Math.trunc(e);switch(t){case"round":return Math.round(e);case"ceil":return Math.ceil(e);case"floor":return Math.floor(e);default:throw new Error(`Unknown roundingMode ${t}`)}}function sn(e){const[t,n,s]=ce(e);return t===1&&n===1&&s===1}function Ha(e,t){return sn(e)||sn(t)}function Ka(e){return ce(e).every(t=>t>0)}function hr(e){if(e==="NHWC")return"channelsLast";if(e==="NCHW")return"channelsFirst";throw new Error(`Unknown dataFormat ${e}`)}function Ja(e,t,n){if(n!=null){if(typeof t=="string")throw Error(`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);if(typeof t=="number")T(Ee(t),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${t}.`);else if(typeof t=="object")t.forEach(s=>{s.forEach(r=>{T(Ee(r),()=>`Error in ${e}: pad must be an integer when using dimRoundingMode ${n} but got pad ${r}.`)})});else throw Error(`Error in ${e}: Unknown padding parameter: ${t}`)}}/**
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
 */function Xa(e,t){const s={x:D(e,"x","reshape","string_or_numeric")},r={shape:t};return x.runKernel(pi,s,r)}const An=W({reshape_:Xa});/**
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
 */function Za(e){const n={x:D(e,"x","sigmoid","float32")};return x.runKernel(wn,n)}const Ya=W({sigmoid_:Za});/**
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
 */function Qa(e,t){let n=D(e,"broadcastTo","x");const s=n.shape;if(te(t),t.length<n.rank)throw new Error(`broadcastTo(): shape.length=${t.length} < input.rank=${n.rank}.`);if(t.length>n.rank){const c=n.shape.slice();for(;c.length<t.length;)c.unshift(1);n=An(n,c)}const r=n.shape,o=Array.from(t);for(let c=t.length-1;c>=0;c--)if(r[c]===t[c])o[c]=1;else if(n.shape[c]!==1)throw new Error(`broadcastTo(): [${s}] cannot be broadcast to [${t}].`);if(o.map((c,h)=>c>1?h:-1).filter(c=>c>=0).length===0)return cr(n);const a={x:n},l={reps:o};return x.runKernel(bi,a,l)}const tl=W({broadcastTo_:Qa});/**
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
 */function el(e,t,n){te(e),n=n||de(t);const s={shape:e,value:t,dtype:n};return x.runKernel(ai,{},s)}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function Zt(e,t){const n=e.length,s=[];for(let r=0;r<n;r++){const o=n-1-r,i=e[o]||1;(t[t.length-1-r]||1)>1&&i===1&&s.unshift(o)}return s}function xn(e,t){const n=[];for(let s=0;s<t.length;s++){const r=e[e.length-s-1],o=t.length-s-1,i=t[o];(r==null||r===1&&i>1)&&n.unshift(o)}return n}function we(e,t){const n=Math.max(e.length,t.length),s=new Array(n);for(let r=0;r<n;r++){let o=e[e.length-r-1];o==null&&(o=1);let i=t[t.length-r-1];if(i==null&&(i=1),o===1)s[n-r-1]=i;else if(i===1)s[n-r-1]=o;else if(o!==i){const a=`Operands could not be broadcast together with shapes ${e} and ${t}.`;throw Error(a)}else s[n-r-1]=o}return s}const ug=Object.freeze(Object.defineProperty({__proto__:null,assertAndGetBroadcastShape:we,getBroadcastDims:Zt,getReductionAxes:xn},Symbol.toStringTag,{value:"Module"}));/**
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
 */function nl(e){const n={x:D(e,"x","zerosLike")};return x.runKernel(Si,n)}const gt=W({zerosLike_:nl});/**
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
 */function sl(e){const n={x:D(e,"x","elu","float32")};return x.runKernel(ii,n)}const rl=W({elu_:sl});/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function vn(e,t){for(let n=0;n<e.length;++n)if(e[e.length-n-1]!==t-1-n)return!1;return!0}function fr(e,t,n){const s=e.length+t.length,r=[];let o=0,i=0;for(let a=0;a<s;a++)n.indexOf(a)===-1?r.push(e[o++]):r.push(t[i++]);return r}function dr(e,t){const n=[],s=e.length;for(let o=0;o<s;o++)t.indexOf(o)===-1&&n.push(e[o]);const r=t.map(o=>e[o]);return[n,r]}function gr(e,t){const n=t.map(s=>1);return fr(e,n,t)}function ol(e,t,n){T(vn(t,n),()=>`${e} supports only inner-most axes for now. Got axes ${t} and rank-${n} input.`)}function pr(e,t){if(vn(e,t))return null;const n=[];for(let s=0;s<t;++s)e.indexOf(s)===-1&&n.push(s);return e.forEach(s=>n.push(s)),n}function il(e){return e.map((t,n)=>[n,t]).sort((t,n)=>t[1]-n[1]).map(t=>t[0])}function mr(e,t){const n=[];for(let s=t-e;s<t;++s)n.push(s);return n}/**
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
 */function al(e,t){let n=D(e,"base","pow"),s=D(t,"exp","pow");[n,s]=vt(n,s);const r={a:n,b:s};return x.runKernel(ui,r)}const Hn=W({pow_:al});/**
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
 */function Pt(e,t){if((at(e)&&t!=="string"||Array.isArray(e))&&t!=="complex64")throw new Error("Error creating a new Scalar: value must be a primitive (number|boolean|string)");if(t==="string"&&at(e)&&!(e instanceof Uint8Array))throw new Error("When making a scalar from encoded string, the value must be `Uint8Array`.");return Ys(e,[],[],t)}/**
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
 */function ll(e){const n={x:D(e,"x","sqrt","float32")};return x.runKernel(yn,n)}const Yt=W({sqrt_:ll});/**
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
 */function cl(e){const t=D(e,"x","square"),n={};return x.runKernel("Square",{x:t},n)}const Dt=W({square_:cl});/**
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
 */function ul(e,t=null,n=!1){let s=D(e,"x","sum");s.dtype==="bool"&&(s=Me(s,"int32"));const r={x:s},o={axis:t,keepDims:n};return x.runKernel(yi,r,o)}const hl=W({sum_:ul});/**
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
 */function fl(e,t=.2){const s={x:D(e,"x","leakyRelu")},r={alpha:t};return x.runKernel(li,s,r)}const dl=W({leakyRelu_:fl});/**
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
 */function hg(e){return T(At(e),()=>"The f passed in grad(f) must be a function"),(t,n)=>{const s=D(t,"x","tf.grad","string_or_numeric"),r=n!=null?D(n,"dy","tf.grad"):null;return x.tidy(()=>{const{value:o,grads:i}=x.gradients(()=>e(s),[s],r);return r!=null&&fe(o.shape,r.shape,"The shape of dy passed in grad(f)(x, dy) must match the shape returned by f(x)"),Pe(i),i[0]})}}function fg(e){return T(At(e),()=>"The f passed in grads(f) must be a function"),(t,n)=>{T(Array.isArray(t),()=>"The args passed in grads(f)(args) must be an array of `Tensor`s or `TensorLike`s");const s=Zi(t,"args","tf.grads","string_or_numeric"),r=n!=null?D(n,"dy","tf.grads"):null;return x.tidy(()=>{const{value:o,grads:i}=x.gradients(()=>e(...s),s,r);return r!=null&&fe(o.shape,r.shape,"The shape of dy passed in grads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Pe(i),i})}}function dg(e){return T(At(e),()=>"The f passed in valueAndGrad(f) must be a function"),(t,n)=>{T(t instanceof Y,()=>"The x passed in valueAndGrad(f)(x) must be a tensor"),T(n==null||n instanceof Y,()=>"The dy passed in valueAndGrad(f)(x, dy) must be a tensor");const{grads:s,value:r}=x.gradients(()=>e(t),[t],n);return Pe(s),{grad:s[0],value:r}}}function gg(e){return T(At(e),()=>"The f passed in valueAndGrads(f) must be a function"),(t,n)=>{T(Array.isArray(t)&&t.every(r=>r instanceof Y),()=>"The args passed in valueAndGrads(f)(args) must be array of tensors"),T(n==null||n instanceof Y,()=>"The dy passed in valueAndGrads(f)(args, dy) must be a tensor");const s=x.gradients(()=>e(...t),t,n);return n!=null&&fe(s.value.shape,n.shape,"The shape of dy passed in valueAndGrads(f)([x1,...], dy) must match the shape returned by f([x1,...])"),Pe(s.grads),s}}function gl(e,t){T(At(e),()=>"The f passed in variableGrads(f) must be a function"),T(t==null||Array.isArray(t)&&t.every(c=>c instanceof ve),()=>"The varList passed in variableGrads(f, varList) must be an array of variables");const n=t!=null;if(!n){t=[];for(const c in x.registeredVariables)t.push(x.registeredVariables[c])}const s=n?t.filter(c=>!c.trainable):null,r=t.length;t=t.filter(c=>c.trainable),T(t.length>0,()=>`variableGrads() expects at least one of the input variables to be trainable, but none of the ${r} variables is trainable.`);const o=!0,{value:i,grads:a}=x.gradients(e,t,null,o);T(a.some(c=>c!=null),()=>"Cannot find a connection between any variable and the result of the loss function y=f(x). Please make sure the operations that use variables are inside the function f passed to minimize()."),T(i.rank===0,()=>`The f passed in variableGrads(f) must return a scalar, but it returned a rank-${i.rank} tensor`);const l={};return t.forEach((c,h)=>{a[h]!=null&&(l[c.name]=a[h])}),s!=null&&s.forEach(c=>l[c.name]=null),{value:i,grads:l}}function pg(e){return x.customGrad(e)}function Pe(e){if(e.filter(n=>n==null).length>0)throw new Error(`Cannot compute gradient of y=f(x) with respect to x. Make sure that
    the f you passed encloses all operations that lead from x to y.`)}/**
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
 */function pl(e,t){let n=D(e,"a","sub"),s=D(t,"b","sub");[n,s]=vt(n,s);const r={a:n,b:s};return x.runKernel(bn,r)}const jt=W({sub_:pl});/**
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
 */function ml(e,t){let n=D(e,"a","maximum"),s=D(t,"b","maximum");[n,s]=vt(n,s),n.dtype==="bool"&&(n=Me(n,"int32"),s=Me(s,"int32")),we(n.shape,s.shape);const r={a:n,b:s};return x.runKernel(pn,r)}const wl=W({maximum_:ml});/**
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
 */function Kn(e,t="float32"){if(te(e),t==="complex64"){const s=Kn(e,"float32"),r=Kn(e,"float32");return Zs(s,r)}const n=pt(F(e),t);return x.makeTensor(n,e,t)}/**
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
 */function yl(e,t){const n=D(e,"x","prelu"),s=D(t,"alpha","prelu"),r={x:n,alpha:s};return x.runKernel(hi,r)}const bl=W({prelu_:yl});/**
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
 */function Il(e){const n={x:D(e,"x","relu")};return x.runKernel(gi,n)}const Sl=W({relu_:Il});/**
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
 */function El(e){const n={x:D(e,"x","relu6")};return x.runKernel(mi,n)}const kl=W({relu6_:El});/**
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
 */function Tl(e,t=0){const s={x:D(e,"x","step")},r={alpha:t};return x.runKernel(Ei,s,r)}const Al=W({step_:Tl});function Mn(e,t,n){const s=t.rank>1?t.shape[t.rank-1]:1,r=t.rank>1?t.rank-1:1,o=`Must have updates.shape = indices.shape[:batchDim] + shape[sliceDim:], got updates.shape: ${n.shape}, indices.shape: ${t.shape}, shape: ${e}, sliceDim: ${s}, and batchDim: ${r}.`;if(n.rank<r)throw new Error(o+` update.rank < ${r}. `);if(e.length<s+(n.rank-r))throw new Error(o+` Output shape length < ${s+(n.rank-r)}`);if(n.rank!==r+e.length-s)throw new Error(o+` update.rank != ${r+e.length-s}`);for(let i=0;i<r;++i)if(n.shape[i]!==t.shape[i])throw new Error(o+` updates.shape[${i}] (${n.shape[i]}) != indices.shape[${i}] (${t.shape[i]}).`);for(let i=0;i<n.rank-r;++i)if(n.shape[i+r]!==e[i+s])throw new Error(o+` updates.shape[${i+r}] (${n.shape[i+r]}) != shape[${i+r}] (${e[i+r]})`)}function wr(e,t,n){if(t.rank<1)throw new Error(`tf.scatterND() expects the indices to be rank 1 or higher, but the rank was ${t.rank}.`);if(e.rank<1)throw new Error(`tf.scatterND() expects the updates to be rank 1 or higher, but the rank was ${e.rank}.`);if(t.dtype!=="int32")throw new Error(`The dtype of 'indices' should be int32, but got dtype: ${t.dtype}`);if(n.length<1)throw new Error(`Output rank must be greater or equal to 1, but got shape: ${n}`);if(n.length===0){if(t.size===0)throw new Error(`Indices specified for empty output. indices shape: ${t.shape}`);if(e.size===0)throw new Error(`Updates specified for empty output. updates shape: ${e.shape}`)}Mn(n,t,e)}function yr(e,t,n){const s=t.shape.length,r=s>1?t.shape[s-1]:1,o=n.length;let i=1;for(let u=r;u<o;++u)i*=n[u];const a=r<1?1:r,l=F(t.shape)/a,c=[...Z(n.slice(0,r)),1],h=F(n);return{sliceRank:r,numUpdates:l,sliceSize:i,strides:c,outputSize:h}}const mg=Object.freeze(Object.defineProperty({__proto__:null,calculateShapes:yr,validateInput:wr,validateUpdateShape:Mn},Symbol.toStringTag,{value:"Module"}));/**
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
 */function wg(e,t){const n=[];for(let o=0;o<t.length;o++)t[o]&&n.push(o);const s=st(e,"int32"),r=st([n.length,e.length],"int32");for(let o=0;o<n.length;o++){const i=s.indexToLoc(n[o]),a=o*e.length;r.values.set(i,a)}return r.toTensor()}/**
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
 */function xl(e,t,n){if(n==null||n==="linear")return e;if(n==="relu")return _(e,Al(t));throw new Error(`Cannot compute gradient for fused activation ${n}.`)}function vl(e,t){let n=t;const s=xn(e.shape,t.shape);return s.length>0&&(n=hl(n,s)),An(n,e.shape)}function Ml(e,t,n,s){if(t==="linear")return e;if(t==="relu")return Sl(e);if(t==="elu")return rl(e);if(t==="relu6")return kl(e);if(t==="prelu")return bl(e,n);if(t==="leakyrelu")return dl(e,s);if(t==="sigmoid")return Ya(e);throw new Error(`Unknown fused activation ${t}.`)}const Fl=(e,t)=>!(e>0)||t==="linear";/**
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
 */function $l(e,t,n){const s=Rl(e,t,n),r=s<0?-(s+1):s;e.splice(r,0,t)}function Rl(e,t,n){return Dl(e,t,n||Nl)}function Nl(e,t){return e>t?1:e<t?-1:0}function Dl(e,t,n){let s=0,r=e.length,o=0,i=!1;for(;s<r;){o=s+(r-s>>>1);const a=n(t,e[o]);a>0?s=o+1:(r=o,i=!a)}return i?s:-s-1}/**
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
 */function yg(e,t,n,s,r){return Fn(e,t,n,s,r,0)}function bg(e,t,n,s,r,o){return Fn(e,t,n,s,r,0,!1,o,!0)}function Ig(e,t,n,s,r,o){return Fn(e,t,n,s,r,o,!0)}function Fn(e,t,n,s,r,o,i=!1,a=!1,l=!1){const c=[];for(let p=0;p<t.length;p++)t[p]>r&&c.push({score:t[p],boxIndex:p,suppressBeginIndex:0});c.sort(Jn);const h=o>0?-.5/o:0,u=[],f=[];for(;u.length<n&&c.length>0;){const p=c.pop(),{score:w,boxIndex:b,suppressBeginIndex:I}=p;if(w<r)break;let k=!1;for(let v=u.length-1;v>=I;--v){const $=_l(e,b,u[v]);if($>=s){k=!0;break}if(p.score=p.score*Bl(s,h,$),p.score<=r)break}p.suppressBeginIndex=u.length,k||(p.score===w?(u.push(b),f.push(p.score)):p.score>r&&$l(c,p,Jn))}const g=u.length,m=n-g;a&&m>0&&(u.push(...new Array(m).fill(0)),f.push(...new Array(m).fill(0)));const y={selectedIndices:u};return i&&(y.selectedScores=f),l&&(y.validOutputs=g),y}function _l(e,t,n){const s=e.subarray(t*4,t*4+4),r=e.subarray(n*4,n*4+4),o=Math.min(s[0],s[2]),i=Math.min(s[1],s[3]),a=Math.max(s[0],s[2]),l=Math.max(s[1],s[3]),c=Math.min(r[0],r[2]),h=Math.min(r[1],r[3]),u=Math.max(r[0],r[2]),f=Math.max(r[1],r[3]),g=(a-o)*(l-i),m=(u-c)*(f-h);if(g<=0||m<=0)return 0;const y=Math.max(o,c),p=Math.max(i,h),w=Math.min(a,u),b=Math.min(l,f),I=Math.max(w-y,0)*Math.max(b-p,0);return I/(g+m-I)}function Bl(e,t,n){const s=Math.exp(t*n*n);return n<=e?s:0}function Jn(e,t){return e.score-t.score||e.score===t.score&&t.boxIndex-e.boxIndex}/**
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
 */const Ol=new Map,rn=new Map;class br{getClassName(){return this.constructor.className}static fromConfig(t,n){return new t(n)}}class St{constructor(){this.classNameMap={}}static getMap(){return St.instance==null&&(St.instance=new St),St.instance}static register(t){St.getMap().classNameMap[t.className]=[t,t.fromConfig]}}function Ir(e,t,n){T(e.className!=null,()=>"Class being registered does not have the static className property defined."),T(typeof e.className=="string",()=>"className is required to be a string, but got type "+typeof e.className),T(e.className.length>0,()=>"Class being registered has an empty-string as its className, which is disallowed."),typeof t>"u"&&(t="Custom"),typeof n>"u"&&(n=e.className);const s=n,r=t+">"+s;return St.register(e),Ol.set(r,e),rn.set(e,r),e}function Pl(e){return rn.has(e)?rn.get(e):e.className}const Sg=Object.freeze(Object.defineProperty({__proto__:null,Serializable:br,SerializationMap:St,getRegisteredName:Pl,registerClass:Ir},Symbol.toStringTag,{value:"Module"}));/**
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
 */class Ct extends br{minimize(t,n=!1,s){const{value:r,grads:o}=this.computeGradients(t,s);if(s!=null){const i=s.map(a=>({name:a.name,tensor:o[a.name]}));this.applyGradients(i)}else this.applyGradients(o);return it(o),n?r:(r.dispose(),null)}get iterations(){return this.iterations_==null&&(this.iterations_=0),this.iterations_}incrementIterations(){this.iterations_=this.iterations+1}computeGradients(t,n){return gl(t,n)}dispose(){this.iterations_!=null&&it(this.iterations_)}async saveIterations(){return this.iterations_==null&&(this.iterations_=0),{name:"iter",tensor:Pt(this.iterations_,"int32")}}async getWeights(){throw new Error("getWeights() is not implemented for this optimizer yet.")}async setWeights(t){throw new Error(`setWeights() is not implemented for this optimizer class ${this.getClassName()}`)}async extractIterations(t){return this.iterations_=(await t[0].tensor.data())[0],t.slice(1)}}Object.defineProperty(Ct,Symbol.hasInstance,{value:e=>e.minimize!=null&&e.computeGradients!=null&&e.applyGradients!=null});/**
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
 */class Ll extends Ct{static get className(){return"Adadelta"}constructor(t,n,s=null){super(),this.learningRate=t,this.rho=n,this.epsilon=s,this.accumulatedGrads=[],this.accumulatedUpdates=[],s==null&&(this.epsilon=x.backend.epsilon())}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=x.registeredVariables[s],i=!1;this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accum_grad`,variable:z(()=>gt(o).variable(i))}),this.accumulatedUpdates[r]==null&&(this.accumulatedUpdates[r]={originalName:`${s}/accum_var`,variable:z(()=>gt(o).variable(i))});const a=Array.isArray(t)?t[r].tensor:t[s];if(a==null)return;const l=this.accumulatedGrads[r].variable,c=this.accumulatedUpdates[r].variable;z(()=>{const h=P(_(l,this.rho),_(Dt(a),1-this.rho)),u=_(dt(Yt(P(c,this.epsilon)),Yt(P(l,this.epsilon))),a),f=P(_(c,this.rho),_(Dt(u),1-this.rho));l.assign(h),c.assign(f);const g=P(_(u,-this.learningRate),o);o.assign(g)})}),this.incrementIterations()}dispose(){this.accumulatedUpdates!=null&&(it(this.accumulatedGrads.map(t=>t.variable)),it(this.accumulatedUpdates.map(t=>t.variable)))}async getWeights(){const t=[...this.accumulatedGrads,...this.accumulatedUpdates];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=t.length/2,s=!1;this.accumulatedGrads=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedUpdates=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,rho:this.rho,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.rho,n.epsilon)}}/**
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
 */class Cl extends Ct{static get className(){return"Adagrad"}constructor(t,n=.1){super(),this.learningRate=t,this.initialAccumulatorValue=n,this.accumulatedGrads=[]}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=x.registeredVariables[s];this.accumulatedGrads[r]==null&&(this.accumulatedGrads[r]={originalName:`${s}/accumulator`,variable:z(()=>el(o.shape,this.initialAccumulatorValue).variable(!1))});const i=Array.isArray(t)?t[r].tensor:t[s];if(i==null)return;const a=this.accumulatedGrads[r].variable;z(()=>{const l=P(a,Dt(i));a.assign(l);const c=P(_(dt(i,Yt(P(l,x.backend.epsilon()))),-this.learningRate),o);o.assign(c)})}),this.incrementIterations()}dispose(){this.accumulatedGrads!=null&&it(this.accumulatedGrads.map(t=>t.variable))}async getWeights(){return[await this.saveIterations()].concat(this.accumulatedGrads.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulatedGrads=t.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,initialAccumulatorValue:this.initialAccumulatorValue}}static fromConfig(t,n){return new t(n.learningRate,n.initialAccumulatorValue)}}/**
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
 */class Ul extends Ct{static get className(){return"Adam"}constructor(t,n,s,r=null){super(),this.learningRate=t,this.beta1=n,this.beta2=s,this.epsilon=r,this.accumulatedFirstMoment=[],this.accumulatedSecondMoment=[],z(()=>{this.accBeta1=Pt(n).variable(),this.accBeta2=Pt(s).variable()}),r==null&&(this.epsilon=x.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);z(()=>{const s=jt(1,this.accBeta1),r=jt(1,this.accBeta2);n.forEach((o,i)=>{const a=x.registeredVariables[o],l=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${o}/m`,variable:z(()=>gt(a).variable(l))}),this.accumulatedSecondMoment[i]==null&&(this.accumulatedSecondMoment[i]={originalName:`${o}/v`,variable:z(()=>gt(a).variable(l))});const c=Array.isArray(t)?t[i].tensor:t[o];if(c==null)return;const h=this.accumulatedFirstMoment[i].variable,u=this.accumulatedSecondMoment[i].variable,f=P(_(h,this.beta1),_(c,1-this.beta1)),g=P(_(u,this.beta2),_(Dt(c),1-this.beta2)),m=dt(f,s),y=dt(g,r);h.assign(f),u.assign(g);const p=P(_(dt(m,P(Yt(y),this.epsilon)),-this.learningRate),a);a.assign(p)}),this.accBeta1.assign(_(this.accBeta1,this.beta1)),this.accBeta2.assign(_(this.accBeta2,this.beta2))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.accBeta2.dispose(),this.accumulatedFirstMoment!=null&&it(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedSecondMoment!=null&&it(this.accumulatedSecondMoment.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedFirstMoment,...this.accumulatedSecondMoment];return[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t),z(()=>{this.accBeta1.assign(Hn(this.beta1,this.iterations_+1)),this.accBeta2.assign(Hn(this.beta2,this.iterations_+1))});const n=t.length/2,s=!1;this.accumulatedFirstMoment=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedSecondMoment=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)}))}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon)}}/**
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
 */class Wl extends Ct{static get className(){return"Adamax"}constructor(t,n,s,r=null,o=0){super(),this.learningRate=t,this.beta1=n,this.beta2=s,this.epsilon=r,this.decay=o,this.accumulatedFirstMoment=[],this.accumulatedWeightedInfNorm=[],z(()=>{this.iteration=Pt(0).variable(),this.accBeta1=Pt(n).variable()}),r==null&&(this.epsilon=x.backend.epsilon())}applyGradients(t){const n=Array.isArray(t)?t.map(s=>s.name):Object.keys(t);z(()=>{const s=jt(1,this.accBeta1),r=dt(-this.learningRate,P(_(this.iteration,this.decay),1));n.forEach((o,i)=>{const a=x.registeredVariables[o],l=!1;this.accumulatedFirstMoment[i]==null&&(this.accumulatedFirstMoment[i]={originalName:`${o}/m`,variable:gt(a).variable(l)}),this.accumulatedWeightedInfNorm[i]==null&&(this.accumulatedWeightedInfNorm[i]={originalName:`${o}/v`,variable:gt(a).variable(l)});const c=Array.isArray(t)?t[i].tensor:t[o];if(c==null)return;const h=this.accumulatedFirstMoment[i].variable,u=this.accumulatedWeightedInfNorm[i].variable,f=P(_(h,this.beta1),_(c,1-this.beta1)),g=_(u,this.beta2),m=Ca(c),y=wl(g,m);h.assign(f),u.assign(y);const p=P(_(dt(r,s),dt(f,P(y,this.epsilon))),a);a.assign(p)}),this.iteration.assign(P(this.iteration,1)),this.accBeta1.assign(_(this.accBeta1,this.beta1))}),this.incrementIterations()}dispose(){this.accBeta1.dispose(),this.iteration.dispose(),this.accumulatedFirstMoment!=null&&it(this.accumulatedFirstMoment.map(t=>t.variable)),this.accumulatedWeightedInfNorm!=null&&it(this.accumulatedWeightedInfNorm.map(t=>t.variable))}async getWeights(){throw new Error("getWeights() is not implemented for Adamax yet.")}async setWeights(t){throw new Error("setWeights() is not implemented for Adamax yet.")}getConfig(){return{learningRate:this.learningRate,beta1:this.beta1,beta2:this.beta2,epsilon:this.epsilon,decay:this.decay}}static fromConfig(t,n){return new t(n.learningRate,n.beta1,n.beta2,n.epsilon,n.decay)}}/**
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
 */class Sr extends Ct{static get className(){return"SGD"}constructor(t){super(),this.learningRate=t,this.setLearningRate(t)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=Array.isArray(t)?t[r].tensor:t[s];if(o==null)return;const i=x.registeredVariables[s];z(()=>{const a=P(_(this.c,o),i);i.assign(a)})}),this.incrementIterations()}setLearningRate(t){this.learningRate=t,this.c!=null&&this.c.dispose(),this.c=ea(Pt(-t))}dispose(){this.c.dispose()}async getWeights(){return[await this.saveIterations()]}async setWeights(t){if(t=await this.extractIterations(t),t.length!==0)throw new Error("SGD optimizer does not have settable weights.")}getConfig(){return{learningRate:this.learningRate}}static fromConfig(t,n){return new t(n.learningRate)}}/**
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
 */class Gl extends Sr{static get className(){return"Momentum"}constructor(t,n,s=!1){super(t),this.learningRate=t,this.momentum=n,this.useNesterov=s,this.accumulations=[],this.m=Pt(this.momentum)}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=x.registeredVariables[s];this.accumulations[r]==null&&(this.accumulations[r]={originalName:`${s}/momentum`,variable:z(()=>gt(o).variable(!1))});const i=this.accumulations[r].variable,a=Array.isArray(t)?t[r].tensor:t[s];a!=null&&z(()=>{let l;const c=P(_(this.m,i),a);this.useNesterov?l=P(_(this.c,P(a,_(c,this.m))),o):l=P(_(this.c,c),o),i.assign(c),o.assign(l)})}),this.incrementIterations()}dispose(){this.m.dispose(),this.accumulations!=null&&it(this.accumulations.map(t=>t.variable))}setMomentum(t){this.momentum=t}async getWeights(){return[await this.saveIterations()].concat(this.accumulations.map(t=>({name:t.originalName,tensor:t.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=!1;this.accumulations=t.map(s=>({originalName:s.name,variable:s.tensor.variable(n)}))}getConfig(){return{learningRate:this.learningRate,momentum:this.momentum,useNesterov:this.useNesterov}}static fromConfig(t,n){return new t(n.learningRate,n.momentum,n.useNesterov)}}/**
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
 */class zl extends Ct{static get className(){return"RMSProp"}constructor(t,n=.9,s=0,r=null,o=!1){if(super(),this.learningRate=t,this.decay=n,this.momentum=s,this.epsilon=r,this.accumulatedMeanSquares=[],this.accumulatedMoments=[],this.accumulatedMeanGrads=[],this.centered=o,r==null&&(this.epsilon=x.backend.epsilon()),t==null)throw new Error("learningRate for RMSPropOptimizer must be defined.")}applyGradients(t){(Array.isArray(t)?t.map(s=>s.name):Object.keys(t)).forEach((s,r)=>{const o=x.registeredVariables[s],i=!1;this.accumulatedMeanSquares[r]==null&&(this.accumulatedMeanSquares[r]={originalName:`${s}/rms`,variable:z(()=>gt(o).variable(i))}),this.accumulatedMoments[r]==null&&(this.accumulatedMoments[r]={originalName:`${s}/momentum`,variable:z(()=>gt(o).variable(i))}),this.accumulatedMeanGrads[r]==null&&this.centered&&(this.accumulatedMeanGrads[r]={originalName:`${s}/mg`,variable:z(()=>gt(o).variable(i))});const a=Array.isArray(t)?t[r].tensor:t[s];if(a==null)return;const l=this.accumulatedMeanSquares[r].variable,c=this.accumulatedMoments[r].variable;z(()=>{const h=P(_(l,this.decay),_(Dt(a),1-this.decay));if(this.centered){const u=this.accumulatedMeanGrads[r].variable,f=P(_(u,this.decay),_(a,1-this.decay)),g=dt(_(a,this.learningRate),Yt(jt(h,P(Dt(f),this.epsilon)))),m=P(_(c,this.momentum),g);l.assign(h),u.assign(f),c.assign(m);const y=jt(o,m);o.assign(y)}else{const u=P(_(l,this.decay),_(Dt(a),1-this.decay)),f=P(_(c,this.momentum),dt(_(a,this.learningRate),Yt(P(u,this.epsilon))));l.assign(u),c.assign(f);const g=jt(o,f);o.assign(g)}})}),this.incrementIterations()}dispose(){this.accumulatedMeanSquares!=null&&it(this.accumulatedMeanSquares.map(t=>t.variable)),this.accumulatedMeanGrads!=null&&this.centered&&it(this.accumulatedMeanGrads.map(t=>t.variable)),this.accumulatedMoments!=null&&it(this.accumulatedMoments.map(t=>t.variable))}async getWeights(){const t=[...this.accumulatedMeanSquares,...this.accumulatedMoments];return this.centered&&t.push(...this.accumulatedMeanGrads),[await this.saveIterations()].concat(t.map(n=>({name:n.originalName,tensor:n.variable})))}async setWeights(t){t=await this.extractIterations(t);const n=this.centered?t.length/3:t.length/2,s=!1;this.accumulatedMeanSquares=t.slice(0,n).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.accumulatedMoments=t.slice(n,n*2).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})),this.centered&&(this.accumulatedMeanGrads=t.slice(n*2,n*3).map(r=>({originalName:r.name,variable:r.tensor.variable(s)})))}getConfig(){return{learningRate:this.learningRate,decay:this.decay,momentum:this.momentum,epsilon:this.epsilon,centered:this.centered}}static fromConfig(t,n){return new t(n.learningRate,n.decay,n.momentum,n.epsilon,n.centered)}}/**
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
 */const ql=[Ll,Cl,Ul,Wl,Gl,zl,Sr];function Vl(){for(const e of ql)Ir(e)}/**
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
 */const jl="model",Hl=".json",Kl=".weights.bin";function Xn(e){return new Promise(t=>setTimeout(t)).then(e)}class Lt{constructor(t){if(!M().getBool("IS_BROWSER"))throw new Error("browserDownloads() cannot proceed because the current environment is not a browser.");t.startsWith(Lt.URL_SCHEME)&&(t=t.slice(Lt.URL_SCHEME.length)),(t==null||t.length===0)&&(t=jl),this.modelJsonFileName=t+Hl,this.weightDataFileName=t+Kl}async save(t){if(typeof document>"u")throw new Error("Browser downloads are not supported in this environment since `document` is not present");const n=mt.join(t.weightData),s=window.URL.createObjectURL(new Blob([n],{type:"application/octet-stream"}));if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserDownloads.save() does not support saving model topology in binary formats yet.");{const r=[{paths:["./"+this.weightDataFileName],weights:t.weightSpecs}],o=tr(t,r),i=window.URL.createObjectURL(new Blob([JSON.stringify(o)],{type:"application/json"})),a=this.modelJsonAnchor==null?document.createElement("a"):this.modelJsonAnchor;if(a.download=this.modelJsonFileName,a.href=i,await Xn(()=>a.dispatchEvent(new MouseEvent("click"))),t.weightData!=null){const l=this.weightDataAnchor==null?document.createElement("a"):this.weightDataAnchor;l.download=this.weightDataFileName,l.href=s,await Xn(()=>l.dispatchEvent(new MouseEvent("click")))}return{modelArtifactsInfo:Oe(t)}}}}Lt.URL_SCHEME="downloads://";class Jl{constructor(t){if(t==null||t.length<1)throw new Error(`When calling browserFiles, at least 1 file is required, but received ${t}`);this.jsonFile=t[0],this.weightsFiles=t.slice(1)}async load(){return new Promise((t,n)=>{const s=new FileReader;s.onload=r=>{const o=JSON.parse(r.target.result),i=o.modelTopology;if(i==null){n(new Error(`modelTopology field is missing from file ${this.jsonFile.name}`));return}if(o.weightsManifest==null){n(new Error(`weightManifest field is missing from file ${this.jsonFile.name}`));return}if(this.weightsFiles.length===0){t({modelTopology:i});return}const l=er(o,c=>this.loadWeights(c));t(l)},s.onerror=r=>n(`Failed to read model topology and weights manifest JSON from file '${this.jsonFile.name}'. BrowserFiles supports loading Keras-style tf.Model artifacts only.`),s.readAsText(this.jsonFile)})}loadWeights(t){const n=[],s=[];for(const i of t)n.push(...i.weights),s.push(...i.paths);const r=this.checkManifestAndWeightFiles(t),o=s.map(i=>this.loadWeightsFile(i,r[i]));return Promise.all(o).then(i=>[n,i])}loadWeightsFile(t,n){return new Promise((s,r)=>{const o=new FileReader;o.onload=i=>{const a=i.target.result;s(a)},o.onerror=i=>r(`Failed to weights data from file of path '${t}'.`),o.readAsArrayBuffer(n)})}checkManifestAndWeightFiles(t){const n=[],s=this.weightsFiles.map(o=>Vn(o.name)),r={};for(const o of t)o.paths.forEach(i=>{const a=Vn(i);if(n.indexOf(a)!==-1)throw new Error(`Duplicate file basename found in weights manifest: '${a}'`);if(n.push(a),s.indexOf(a)===-1)throw new Error(`Weight file with basename '${a}' is not provided.`);r[i]=this.weightsFiles[s.indexOf(a)]});if(n.length!==this.weightsFiles.length)throw new Error(`Mismatch in the number of files in weights manifest (${n.length}) and the number of weight files provided (${this.weightsFiles.length}).`);return r}}const Xl=e=>M().getBool("IS_BROWSER")&&!Array.isArray(e)&&e.startsWith(Lt.URL_SCHEME)?Zl(e.slice(Lt.URL_SCHEME.length)):null;U.registerSaveRouter(Xl);function Zl(e="model"){return new Lt(e)}function Eg(e){return new Jl(e)}/**
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
 */function Zn(e,t,n,s){i(e),n=n??0,s=s??1,a(n,s);let r=0;const o=l=>(l.then(c=>{const h=n+ ++r/e.length*(s-n);return t(h),c}),l);function i(l){T(l!=null&&Array.isArray(l)&&l.length>0,()=>"promises must be a none empty array")}function a(l,c){T(l>=0&&l<=1,()=>`Progress fraction must be in range [0, 1], but got startFraction ${l}`),T(c>=0&&c<=1,()=>`Progress fraction must be in range [0, 1], but got endFraction ${c}`),T(c>=l,()=>`startFraction must be no more than endFraction, but got startFraction ${l} and endFraction ${c}`)}return Promise.all(e.map(o))}/**
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
 */async function Er(e,t){t==null&&(t={});const n=t.fetchFunc==null?M().platform.fetch:t.fetchFunc,s=e.map(u=>n(u,t.requestInit,{isBinary:!0})),a=(t.onProgress==null?await Promise.all(s):await Zn(s,t.onProgress,0,.5)).map(u=>u.arrayBuffer());return t.onProgress==null?await Promise.all(a):await Zn(a,t.onProgress,.5,1)}function Yl(e,t){var n;const s=t.fetchFunc==null?M().platform.fetch:t.fetchFunc;let r=0,o;return(n=t.onProgress)===null||n===void 0||n.call(t,0),new ReadableStream({pull:async i=>{for(var a;r<e.length;){o||(o=(await s(e[r],t.requestInit,{isBinary:!0})).body.getReader());const{done:l,value:c}=await o.read();if(l){r++,o=void 0,(a=t.onProgress)===null||a===void 0||a.call(t,r/e.length);continue}i.enqueue(c);return}i.close()}})}async function kg(e,t="",n,s){return Ql(i=>Er(i,{requestInit:s}))(e,t,n)}function Ql(e){return async(t,n="",s)=>{const r=t.map(()=>!1),o={},i=s!=null?s.map(()=>!1):[],a=[];if(t.forEach((g,m)=>{let y=0;g.weights.forEach(p=>{const w="quantization"in p?p.quantization.dtype:p.dtype,b=_t[w]*F(p.shape),I=()=>{r[m]=!0,o[m]==null&&(o[m]=[]),o[m].push({manifestEntry:p,groupOffset:y,sizeBytes:b})};s!=null?s.forEach((k,v)=>{k===p.name&&(I(),i[v]=!0)}):I(),a.push(p.name),y+=b})}),!i.every(g=>g)){const g=s.filter((m,y)=>!i[y]);throw new Error(`Could not find weights in manifest with names: ${g.join(", ")}. 
Manifest JSON has weights with names: ${a.join(", ")}.`)}const l=r.reduce((g,m,y)=>(m&&g.push(y),g),[]),c=[];l.forEach(g=>{t[g].paths.forEach(m=>{const y=n+(n.endsWith("/")?"":"/")+m;c.push(y)})});const h=await e(c),u={};let f=0;return l.forEach(g=>{const m=t[g].paths.length,y=new mt(h.slice(f,f+m));o[g].forEach(w=>{const b=y.slice(w.groupOffset,w.groupOffset+w.sizeBytes),I=ra(b,[w.manifestEntry]);for(const k in I)u[k]=I[k]}),f+=m}),u}}/**
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
 */const tc="application/octet-stream",ec="application/json";class $n{constructor(t,n){if(this.DEFAULT_METHOD="POST",n==null&&(n={}),this.weightPathPrefix=n.weightPathPrefix,this.weightUrlConverter=n.weightUrlConverter,n.fetchFunc!=null?(T(typeof n.fetchFunc=="function",()=>"Must pass a function that matches the signature of `fetch` (see https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)"),this.fetch=n.fetchFunc):this.fetch=M().platform.fetch,T(t!=null&&t.length>0,()=>"URL path for http must not be null, undefined or empty."),Array.isArray(t)&&T(t.length===2,()=>`URL paths for http must have a length of 2, (actual length is ${t.length}).`),this.path=t,n.requestInit!=null&&n.requestInit.body!=null)throw new Error("requestInit is expected to have no pre-existing body, but has one.");this.requestInit=n.requestInit||{},this.loadOptions=n}async save(t){if(t.modelTopology instanceof ArrayBuffer)throw new Error("BrowserHTTPRequest.save() does not support saving model topology in binary formats yet.");const n=Object.assign({method:this.DEFAULT_METHOD},this.requestInit);n.body=new FormData;const s=[{paths:["./model.weights.bin"],weights:t.weightSpecs}],r=tr(t,s);if(n.body.append("model.json",new Blob([JSON.stringify(r)],{type:ec}),"model.json"),t.weightData!=null){const i=mt.join(t.weightData);n.body.append("model.weights.bin",new Blob([i],{type:tc}),"model.weights.bin")}const o=await this.fetch(this.path,n);if(o.ok)return{modelArtifactsInfo:Oe(t),responses:[o]};throw new Error(`BrowserHTTPRequest.save() failed due to HTTP response status ${o.status}.`)}async loadModelJSON(){const t=await this.fetch(this.path,this.requestInit);if(!t.ok)throw new Error(`Request to ${this.path} failed with status code ${t.status}. Please verify this URL points to the model JSON of the model to load.`);let n;try{n=await t.json()}catch{let i=`Failed to parse model JSON of response from ${this.path}.`;throw this.path.endsWith(".pb")?i+=" Your path contains a .pb file extension. Support for .pb models have been removed in TensorFlow.js 1.0 in favor of .json models. You can re-convert your Python TensorFlow model using the TensorFlow.js 1.0 conversion scripts or you can convert your.pb models with the 'pb2json'NPM script in the tensorflow/tfjs-converter repository.":i+=" Please make sure the server is serving valid JSON for this request.",new Error(i)}const s=n.modelTopology,r=n.weightsManifest;if(s==null&&r==null)throw new Error(`The JSON from HTTP path ${this.path} contains neither model topology or manifest for weights.`);return n}async load(){if(this.loadOptions.streamWeights)return this.loadStream();const t=await this.loadModelJSON();return er(t,n=>this.loadWeights(n))}async loadStream(){const t=await this.loadModelJSON(),n=await this.getWeightUrls(t.weightsManifest),s=jn(t.weightsManifest),r=()=>Yl(n,this.loadOptions);return Object.assign(Object.assign({},t),{weightSpecs:s,getWeightStream:r})}async getWeightUrls(t){const n=Array.isArray(this.path)?this.path[1]:this.path,[s,r]=nc(n),o=this.weightPathPrefix||s,i=[],a=[];for(const l of t)for(const c of l.paths)this.weightUrlConverter!=null?a.push(this.weightUrlConverter(c)):i.push(o+c+r);return this.weightUrlConverter&&i.push(...await Promise.all(a)),i}async loadWeights(t){const n=await this.getWeightUrls(t),s=jn(t),r=await Er(n,this.loadOptions);return[s,r]}}$n.URL_SCHEME_REGEX=/^https?:\/\//;function nc(e){const t=e.lastIndexOf("/"),n=e.lastIndexOf("?"),s=e.substring(0,t),r=n>t?e.substring(n):"";return[s+"/",r]}function Yn(e){return e.match($n.URL_SCHEME_REGEX)!=null}const kr=(e,t)=>{if(typeof fetch>"u"&&(t==null||t.fetchFunc==null))return null;{let n=!0;if(Array.isArray(e)?n=e.every(s=>Yn(s)):n=Yn(e),n)return Tr(e,t)}return null};U.registerSaveRouter(kr);U.registerLoadRouter(kr);function Tr(e,t){return new $n(e,t)}function Tg(e,t){return Tr(e,t)}function Ar(e,t){const n=e.shape.length,s=t.shape.length;if(n<1)throw new Error(`tf.gatherND() expects the input to be rank 1 or higher, but the rank was ${n}.`);if(s<1)throw new Error(`tf.gatherND() expects the indices to be rank 1 or higher, but the rank was ${s}.`);if(t.dtype!=="int32")throw new Error(`tf.gatherND() expects the indices to be int32 type, but the dtype was ${t.dtype}.`);if(t.shape[s-1]>n)throw new Error(`index innermost dimension length must be <= tensor rank; saw: ${t.shape[s-1]} vs. ${n}`);if(F(e.shape)===0)throw new Error(`Requested more than 0 entries, but input is empty. Input shape: ${e.shape}.`);const r=t.shape,o=r[r.length-1];let i=1;for(let u=0;u<r.length-1;++u)i*=r[u];const a=e.shape,l=r.slice();l.pop();let c=1;for(let u=o;u<n;++u)c*=a[u],l.push(a[u]);const h=[...Z(e.shape).map(u=>u/c),1].slice(0,o);return[l,i,c,h]}const Ag=Object.freeze(Object.defineProperty({__proto__:null,prepareAndValidate:Ar},Symbol.toStringTag,{value:"Module"}));/**
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
 */const on=-2,sc=-1;function xr(e,t,n){const s=e.shape.length;T(s===t.length,()=>`Error in slice${s}D: Length of begin ${t} must match the rank of the array (${s}).`),T(s===n.length,()=>`Error in slice${s}D: Length of size ${n} must match the rank of the array (${s}).`);for(let r=0;r<s;++r)T(t[r]+n[r]<=e.shape[r],()=>`Error in slice${s}D: begin[${r}] + size[${r}] (${t[r]+n[r]}) would overflow input.shape[${r}] (${e.shape[r]})`)}function rc(e){const t=[];let n=0;for(;e>0;)e&1&&t.push(n),e/=2,n++;return t}function oc(e,t,n){const s=[];for(let r=0;r<e.length;r++)s[r]=Math.ceil((t[r]-e[r])/n[r]);return s}function vr(e,t,n,s){const r=[...e];for(let o=r.length;o<s.length;o++)r.push(1);for(let o=0;o<n;o++)o===0?r[t]=1:(r.splice(t,0,1),r.pop());return r}function Mr(e,t,n){return n<=e?n:n-(t-1)}function Fr(e,t){const n=[];for(let s=0;s<e;s++)n.push(t+s);return n}function ic(e,t,n,s,r,o,i,a,l){const c=e.length;let h=new Array(c),u=new Array(c),f=new Array(c);if(t.length&&n>0){const g=t[0],m=n+1;h=$r(i,g,m,s,e),u=Rr(a,g,m,r,e),f=vr(o,g,m,e)}else for(let g=0;g<c;g++)h[g]=Dr(i,s,o,e,g,l),u[g]=_r(a,r,o,e,g,l),f[g]=Nr(o,g,l);return{begin:h,end:u,strides:f}}function $r(e,t,n,s,r){const o=[...r],i=Fr(n,t);for(let a=0;a<o.length;a++)if(i.indexOf(a)>-1)o[a]=0;else{const l=Mr(t,n,a);let c=s[l];e&1<<l&&(c=0),o[a]=c}return o}function Rr(e,t,n,s,r){const o=[...r],i=Fr(n,t);for(let a=0;a<o.length;a++)if(i.indexOf(a)>-1)o[a]=Number.MAX_SAFE_INTEGER;else{const l=Mr(t,n,a);let c=s[l];e&1<<l&&(c=Number.MAX_SAFE_INTEGER),o[a]=c}for(let a=0;a<o.length;a++){const l=r[a];o[a]<0&&(o[a]+=l),o[a]=ie(0,o[a],r[a])}return o}function Nr(e,t,n){let s=e[t];return(n&1<<t||s==null)&&(s=1),s}function Dr(e,t,n,s,r,o){let i=t[r];const a=n[r]||1;(e&1<<r||o&1<<r||i==null)&&(a>0?i=Number.MIN_SAFE_INTEGER:i=Number.MAX_SAFE_INTEGER);const l=s[r];return i<0&&(i+=l),i=ie(0,i,l-1),i}function _r(e,t,n,s,r,o){let i=t[r];const a=n[r]||1;(e&1<<r||o&1<<r||i==null)&&(a>0?i=Number.MAX_SAFE_INTEGER:i=Number.MIN_SAFE_INTEGER);const l=s[r];return i<0&&(i+=l),a>0?i=ie(0,i,l):i=ie(-1,i,l-1),i}function Br(e,t,n){let s=n.length;for(let r=0;r<n.length;r++)if(n[r]>1){s=r;break}for(let r=s+1;r<n.length;r++)if(t[r]>0||n[r]!==e[r])return!1;return!0}function Or(e,t){let n=e.length>0?e[e.length-1]:1;for(let s=0;s<e.length-1;s++)n+=e[s]*t[s];return n}function Pr(e,t,n){let s;const r=e.shape.length;typeof t=="number"?s=[t,...new Array(r-1).fill(0)]:t.length<r?s=t.concat(new Array(r-t.length).fill(0)):s=t.slice(),s.forEach(i=>{T(i!==-1,()=>"slice() does not support negative begin indexing.")});let o;return n==null?o=new Array(r).fill(-1):typeof n=="number"?o=[n,...new Array(r-1).fill(-1)]:n.length<r?o=n.concat(new Array(r-n.length).fill(-1)):o=n,o=o.map((i,a)=>i>=0?i:(T(i===-1,()=>`Negative size values should be exactly -1 but got ${i} for the slice() size at index ${a}.`),e.shape[a]-s[a])),[s,o]}function ac(e,t,n,s,r,o,i,a,l){let c;if(s==null?(c=new Array(t.length),c.fill(1)):c=s,i!=null&&(i&i-1)!==0)throw new Error("Multiple ellipses in slice is not allowed.");let h=!1;const u={dims:c.length,numAddAxisAfterEllipsis:0,begin:t.slice(),end:n.slice(),strides:c.slice(),beginMask:r,endMask:o,ellipsisMask:i,newAxisMask:a,shrinkAxisMask:l};for(let I=0;I<u.dims;I++)h&&(1<<I&a)!==0&&u.numAddAxisAfterEllipsis++,1<<I&i&&(h=!0);h||(u.ellipsisMask|=1<<u.dims,u.dims++);const f={dims:e.length,beginMask:0,endMask:0,beginValid:!1,endValid:!1};lc(u,f);let g=!0,m=!0,y=!0;const p=[],w=[];for(let I=0;I<e.length;++I){if(f.strides[I]===0)throw Error(`strides[${I}] must be non-zero`);const k=!!(f.shrinkAxisMask&1<<I),v=e[I];if(v===-1){p.push(k?1:-1);continue}const $=[f.beginMask&1<<I,f.endMask&1<<I],B=[f.strides[I]>0?0:-1,f.strides[I]>0?v:v-1];if(k&&f.strides[I]<=0)throw Error("only stride 1 allowed on non-range indexing.");y=y&&f.strides[I]===1;const R=!!(f.beginMask&1<<I&&f.endMask&1<<I);if(f.beginValid&&f.endValid){if(k){const A=f.begin[I]<0?v+f.begin[I]:f.begin[I];if(f.begin[I]=A,f.end[I]=f.begin[I]+1,A<0||A>=v)throw Error(`slice index ${f.begin[I]} of dimension ${I} out of bounds.`)}else f.begin[I]=Qn(f.begin[I],0,f.strides[I],v,$,B),f.end[I]=Qn(f.end[I],1,f.strides[I],v,$,B);const d=f.strides[I]===1&&f.begin[I]===0&&f.end[I]===v;g=g&&d,m=m&&(I===0&&f.strides[I]===1||d)}else g=g&&f.strides[I]===1&&R,m=m&&(I===0&&f.strides[I]===1||R);let E,S=!1;if(f.beginValid&&f.endValid?(E=f.end[I]-f.begin[I],S=!0):k?(E=1,S=!0):R&&v>=0&&(f.strides[I]<0?E=-v:E=v,S=!0),S){let d;E===0||E<0!=f.strides[I]<0?d=0:d=Math.trunc(E/f.strides[I])+(E%f.strides[I]!==0?1:0),p.push(d)}else p.push(-1)}for(let I=0;I<f.finalShapeGatherIndices.length;++I){const k=f.finalShapeGatherIndices[I];k>=0?w.push(p[k]):k===on&&w.push(1)}return{finalShapeSparse:w.filter((I,k)=>f.finalShapeGatherIndices[k]!==on),finalShape:w,isIdentity:g,sliceDim0:m,isSimpleSlice:y,begin:f.begin,end:f.end,strides:f.strides}}function lc(e,t){t.beginMask=0,t.endMask=0,t.shrinkAxisMask=0;let n=0;t.beginValid=e.begin!=null,t.endValid=e.end!=null,t.begin=new Array(t.dims),t.end=new Array(t.dims),t.strides=new Array(t.dims),t.finalShapeGatherIndices=[],t.finalShapeGatherIndicesSparse=[],t.inputShapeGatherIndicesSparse=new Array(t.dims);for(let s=0;s<e.dims;s++)if(1<<s&e.ellipsisMask){const r=Math.min(t.dims-(e.dims-s)+1+e.numAddAxisAfterEllipsis,t.dims);for(;n<r;n++)t.begin[n]=0,t.end[n]=0,t.strides[n]=1,t.beginMask|=1<<n,t.endMask|=1<<n,t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(-1),t.inputShapeGatherIndicesSparse[n]=s}else if(1<<s&e.newAxisMask)t.finalShapeGatherIndices.push(on),t.finalShapeGatherIndicesSparse.push(-1);else{if(n===t.begin.length)throw Error(`Index out of range using input dim ${n}; input has only ${t.dims} dims, ${t.begin.length}.`);e.begin!=null&&(t.begin[n]=e.begin[s]),e.end!=null&&(t.end[n]=e.end[s]),t.strides[n]=e.strides[s],e.beginMask&1<<s&&(t.beginMask|=1<<n),e.endMask&1<<s&&(t.endMask|=1<<n),e.shrinkAxisMask&1<<s?(t.finalShapeGatherIndices.push(sc),t.finalShapeGatherIndicesSparse.push(-1),t.shrinkAxisMask|=1<<n):(t.finalShapeGatherIndices.push(n),t.finalShapeGatherIndicesSparse.push(s)),t.inputShapeGatherIndicesSparse[n]=s,n++}}function Qn(e,t,n,s,r,o){if(r[t])return n>0?o[t]:o[t+1&1];{const i=e<0?s+e:e;return i<o[0]?o[0]:i>o[1]?o[1]:i}}const cc=Object.freeze(Object.defineProperty({__proto__:null,assertParamsValid:xr,computeFlatOffset:Or,computeOutShape:oc,getNormalizedAxes:ic,isSliceContinous:Br,maskToAxes:rc,parseSliceParams:Pr,sliceInfo:ac,startForAxis:Dr,startIndicesWithElidedDims:$r,stopForAxis:_r,stopIndicesWithElidedDims:Rr,stridesForAxis:Nr,stridesWithElidedDims:vr},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function uc(e,t){const n=e[0].length;e.forEach((r,o)=>{T(r.length===n,()=>`Error in concat${n}D: rank of tensors[${o}] must be the same as the rank of the rest (${n})`)}),T(t>=0&&t<n,()=>`Error in concat${n}D: axis must be between 0 and ${n-1}.`);const s=e[0];e.forEach((r,o)=>{for(let i=0;i<n;i++)T(i===t||r[i]===s[i],()=>`Error in concat${n}D: Shape of tensors[${o}] (${r}) does not match the shape of the rest (${s}) along the non-concatenated axis ${o}.`)})}function hc(e,t){const n=e[0].slice();for(let s=1;s<e.length;s++)n[t]+=e[s][t];return n}/**
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
 */var ft;(function(e){e[e.FIRST_DIM_SIZE=0]="FIRST_DIM_SIZE",e[e.VALUE_ROWIDS=1]="VALUE_ROWIDS",e[e.ROW_LENGTHS=2]="ROW_LENGTHS",e[e.ROW_SPLITS=3]="ROW_SPLITS",e[e.ROW_LIMITS=4]="ROW_LIMITS",e[e.ROW_STARTS=5]="ROW_STARTS"})(ft||(ft={}));function Lr(e,t,n){let s=new Array;if(n==null&&t==null)return s;if(t==null)for(;s.length<e+n.length;)s.push(-1);else s=t.slice();if(n==null)return s;if(e+n.length!==s.length)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.rank = ${e+n.length}, but shape.rank = ${s.length}`);for(let r=1;r<n.length;++r){const o=n[r],i=s[s.length-n.length+r],a=s[i];if(o>=0)if(a>=0){if(a!==o)throw new Error(`rt input.shape and shape=${t} are incompatible: rt input.shape[${r+e}] = ${o} but shape[${r+e}] = ${a}`)}else s[i]=o}return s}function Cr(e){const t={FIRST_DIM_SIZE:ft.FIRST_DIM_SIZE,VALUE_ROWIDS:ft.VALUE_ROWIDS,ROW_LENGTHS:ft.ROW_LENGTHS,ROW_SPLITS:ft.ROW_SPLITS,ROW_LIMITS:ft.ROW_LIMITS,ROW_STARTS:ft.ROW_STARTS},n=[];for(const s of e)if(s in t)n.push(t[s]);else break;return n}function Ur(e){return e.length===0?0:e[0]===ft.FIRST_DIM_SIZE?e.length-1:e.length}function Wr(e,t){if(e==null||t==null)return;const n=e.length,s=t.length;if(n>=s)throw new Error(`defaultValue.shape=${e} and ragged tensor flatValues.shape=${t}, are incompatible: defaultValue.rank = ${n} must be less than ragged tensor input flatValues.rank = ${s})`);for(let r=0;r<Math.min(n,s-1);++r){const o=e[r],i=t[r+1];if(o>=0&&i>=0&&o!==1&&o!==i)throw new Error(`defaultValue.shape=${e}, and ragged tensor input flatValues.shape=${t} are incompatible: defaultValue.shape[${r-e.length}] = ${o} but ragged tensor input.flatValues.shape[${r-e.length}] = ${i}`)}}/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Rn=30;function fc(e){return e<=Rn?e:Te(e,Math.floor(Math.sqrt(e)))}/**
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
 */function dc(e,t,n){const s=n*(typeof e=="number"?e:e[0]),r=t*(typeof e=="number"?e:e[1]);return[s,r]}/**
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
 */function gc(e,t,n,s=!0){let r=[];if(s)r=r.concat(t.slice(0)),r.push(e[0]/n),r=r.concat(e.slice(1));else{r=r.concat(e[0]);const o=t.length;for(let i=0;i<o;++i)r=r.concat([e[i+1]/t[i],t[i]]);r=r.concat(e.slice(o+1))}return r}function pc(e,t,n=!0){const s=[];if(n){s.push(t);for(let r=t+1;r<e;++r)r<=2*t?(s.push(r),s.push(r-(t+1))):s.push(r)}else{const r=[],o=[];for(let i=1;i<e;++i)i>=t*2+1||i%2===1?o.push(i):r.push(i);s.push(...r),s.push(0),s.push(...o)}return s}function mc(e,t,n,s=!0){const r=[];s?r.push(e[0]/n):r.push(e[0]*n);for(let o=1;o<e.length;++o)o<=t.length?s?r.push(t[o-1]*e[o]):r.push(e[o]/t[o-1]):r.push(e[o]);return r}function wc(e,t){const n=[0];for(let s=0;s<t;++s)n.push(e[s][0]);return n}function yc(e,t,n){const s=e.slice(0,1);for(let r=0;r<n;++r)s.push(e[r+1]-t[r][0]-t[r][1]);return s}/**
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
 */const bc=1.7580993408473768,Ic=1.0507009873554805;/**
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
 */const Sc=.3275911,Ec=.254829592,kc=-.284496736,Tc=1.421413741,Ac=-1.453152027,xc=1.061405429;/**
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
 */function an(e,t){if(e.length!==t.length)throw new Error(`Cannot merge real and imag arrays of different lengths. real:${e.length}, imag: ${t.length}.`);const n=new Float32Array(e.length*2);for(let s=0;s<n.length;s+=2)n[s]=e[s/2],n[s+1]=t[s/2];return n}function vc(e){const t=new Float32Array(e.length/2),n=new Float32Array(e.length/2);for(let s=0;s<e.length;s+=2)t[s/2]=e[s],n[s/2]=e[s+1];return{real:t,imag:n}}function Mc(e){const t=Math.ceil(e.length/4),n=new Float32Array(t),s=new Float32Array(t);for(let r=0;r<e.length;r+=4)n[Math.floor(r/4)]=e[r],s[Math.floor(r/4)]=e[r+1];return{real:n,imag:s}}function Fc(e){const t=Math.floor(e.length/4),n=new Float32Array(t),s=new Float32Array(t);for(let r=2;r<e.length;r+=4)n[Math.floor(r/4)]=e[r],s[Math.floor(r/4)]=e[r+1];return{real:n,imag:s}}function $c(e,t){const n=e[t*2],s=e[t*2+1];return{real:n,imag:s}}function Rc(e,t,n,s){e[s*2]=t,e[s*2+1]=n}function Nc(e,t){const n=new Float32Array(e/2),s=new Float32Array(e/2);for(let r=0;r<Math.ceil(e/2);r++){const o=(t?2:-2)*Math.PI*(r/e);n[r]=Math.cos(o),s[r]=Math.sin(o)}return{real:n,imag:s}}function Dc(e,t,n){const s=(n?2:-2)*Math.PI*(e/t),r=Math.cos(s),o=Math.sin(s);return{real:r,imag:o}}/**
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
 */const Ve="->",_c=/->/g,ts=",",es="...";function Bc(e,t){e=e.replace(/\s/g,"");const n=(e.length-e.replace(_c,"").length)/Ve.length;if(n<1)throw new Error("Equations without an arrow are not supported.");if(n>1)throw new Error(`Equation must contain exactly one arrow ("${Ve}").`);const[s,r]=e.split(Ve);T(s.indexOf(es)===-1,()=>`The ellipsis notation ("${es}") is not supported yet.`);const o=s.split(ts),i=o.length;if(t!==i)throw new Error(`Expected ${i} input tensors, received ${t}`);if(i>2)throw new Error("Support for more than 2 input tensors is not implemented yet.");const a=[];for(let f=0;f<r.length;++f){const g=r[f];if(!o.some(m=>m.indexOf(g)!==-1))throw new Error(`Output subscripts contain the label ${g} not present in the input subscripts.`);a.indexOf(g)===-1&&a.push(g)}for(let f=0;f<s.length;++f){const g=s[f];a.indexOf(g)===-1&&g!==ts&&a.push(g)}const l=new Array(o.length);for(let f=0;f<i;++f){if(new Set(o[f].split("")).size!==o[f].length)throw new Error(`Found duplicate axes in input component ${o[f]}. Support for duplicate axes in input is not implemented yet.`);l[f]=[];for(let g=0;g<o[f].length;++g)l[f].push(a.indexOf(o[f][g]))}const c=a.length,h=r.length,u=[];for(let f=h;f<c;++f)u.push(f);return{allDims:a,summedDims:u,idDims:l}}function Oc(e,t){let n=new Array(e);n.fill(-1);for(let r=0;r<t.length;++r)n[t[r]]=r;const s=[];for(let r=0;r<e;++r)n[r]===-1&&s.push(r);return n=n.filter(r=>r!==-1),{permutationIndices:n,expandDims:s}}function Pc(e,t,n){const s=new Array(e);for(let r=0;r<n.length;++r){const o=n[r].shape;for(let i=0;i<t[r].length;++i)s[t[r][i]]===void 0?s[t[r][i]]=o[i]:T(s[t[r][i]]===o[i],()=>`Expected dimension ${s[t[r][i]]} at axis ${i} of input shaped ${JSON.stringify(o)}, but got dimension ${o[i]}`)}}function Lc(e,t){const n=e,s=[];let r=0;e.length===0&&n.push(-1),r=e.length+1;for(let i=0;i<r;++i)s.push([]);const o=[];for(let i=0;i<n.length;++i){const a=n[i],l=Uc(t,a);for(const c of l)o.indexOf(c)===-1&&(s[i].push(c),o.push(c))}return{path:n,steps:s}}function Cc(e){return e.every((t,n)=>t===n)}function Uc(e,t){const n=[];for(let s=0;s<e.length;++s)(e[s].length===0||e[s].indexOf(t)!==-1||t===-1)&&n.push(s);return n}function Wc(e,t,n=0){let s=[];if(typeof t=="number")T(e.shape[n]%t===0,()=>"Number of splits must evenly divide the axis."),s=new Array(t).fill(e.shape[n]/t);else{const r=t.reduce((i,a)=>(a===-1&&(i+=1),i),0);T(r<=1,()=>"There should be only one negative value in split array.");const o=t.indexOf(-1);if(o!==-1){const i=t.reduce((a,l)=>l>0?a+l:a);t[o]=e.shape[n]-i}T(e.shape[n]===t.reduce((i,a)=>i+a),()=>"The sum of sizes must match the size of the axis dimension."),s=t}return s}/**
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
 */function Gr(e){return`Received SparseTensor with denseShape[0] = 0 but
  indices.shape[0] = ${e}`}function zr(e,t){return`indices(${e}, 0) is invalid: ${t} < 0`}function qr(e,t,n){return`indices(${e}, 0) is invalid: ${t} >= ${n}`}/**
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
 */function Vr(e,t){return`only one output dimension may be -1, not both ${e} and ${t}`}function jr(e,t){return`size ${e} must be non-negative, not ${t}`}function Hr(){return"reshape cannot infer the missing input size for an empty tensor unless all specified input sizes are non-zero"}function Kr(e,t){const n=F(e),s=F(t);return`Input to reshape is a SparseTensor with ${n}
  dense values, but the requested shape requires a multiple of ${s}. inputShape=${e} outputShape= ${t}`}function Jr(e,t){const n=F(e),s=F(t);return`Input to reshape is a tensor with ${n} dense values, but the requested shape has ${s}. inputShape=${e} outputShape=${t}`}/**
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
 */function ln(){return"segment ids must be >= 0"}function Xr(){return"segment ids are not increasing"}function Zr(e,t){return`Segment id ${e} out of range [0, ${t}), possibly because segmentIds input is not sorted.`}function Yr(e,t,n){return`Bad: indices[${e}] == ${t} out of range [0, ${n})`}/**
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
 */function Gc(e,t){let n=!1,s;for(e<=Rn?(s=e,n=!0):s=Te(e,Math.floor(Math.sqrt(e)));!n;)s>t||s===e?n=!0:s=Te(e,s+1);return s}function zc(e,t,n){const s=[],r=e.length;for(let o=0;o<r;o++)o!==t?s.push(e[o]):s.push(n);return s}function qc(e,t,n,s){const r=t.shape.length,o=e.shape.length;if(s!==0&&(s<-r||s>r))throw new Error(`Expect batchDims in the range of [-${r}, ${r}], but got ${s}`);if(s<0&&(s+=r),s>o)throw new Error(`batchDims (${s}) must be less than rank(x) (
    ${o}).`);if(n<s)throw new Error(`batchDims (${s}) must be less than or equal to axis (${n}).`);for(let u=0;u<s;++u)if(e.shape[u]!==t.shape[u])throw new Error(`x.shape[${u}]: ${e.shape[u]} should be equal to indices.shape[${u}]: ${t.shape[u]}.`);const i=e.shape[n],a=[];let l=1,c=1,h=1;for(let u=0;u<s;++u)a.push(e.shape[u]),l*=e.shape[u];for(let u=s;u<n;u++)a.push(e.shape[u]),c*=e.shape[u];for(let u=s;u<r;u++)a.push(t.shape[u]);for(let u=n+1;u<o;u++)a.push(e.shape[u]),h*=e.shape[u];return{batchSize:l,sliceSize:h,outerSize:c,dimSize:i,outputShape:a}}const Vc=Object.freeze(Object.defineProperty({__proto__:null,collectGatherOpShapeInfo:qc,computeOutShape:zc,segOpComputeOptimalWindowSize:Gc},Symbol.toStringTag,{value:"Module"}));/**
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
 */function Qt(e){try{return e.map(t=>Ae(t))}catch(t){throw new Error(`Failed to decode encoded string bytes into utf-8, error: ${t}`)}}function Qr(e){return e.map(t=>Nt(t))}const xg=Object.freeze(Object.defineProperty({__proto__:null,ERF_A1:Ec,ERF_A2:kc,ERF_A3:Tc,ERF_A4:Ac,ERF_A5:xc,ERF_P:Sc,PARALLELIZE_THRESHOLD:Rn,get RowPartitionType(){return ft},SELU_SCALE:Ic,SELU_SCALEALPHA:bc,applyActivation:Ml,assertAndGetBroadcastShape:we,assertAxesAreInnerMostDims:ol,assertParamsConsistent:uc,assignToTypedArray:Rc,axesAreInnerMostDims:vn,calculateShapes:yr,checkEinsumDimSizes:Pc,checkPadOnDimRoundingMode:Ja,combineLocations:fr,combineRaggedTensorToTensorShapes:Lr,complexWithEvenIndex:Mc,complexWithOddIndex:Fc,computeConv2DInfo:kn,computeConv3DInfo:ur,computeDefaultPad:Tn,computeDilation2DInfo:Ua,computeOptimalWindowSize:fc,computeOutAndReduceShapes:dr,computeOutShape:hc,computePool2DInfo:Wa,computePool3DInfo:Ga,convertConv2DDataFormat:hr,decodeEinsumEquation:Bc,eitherStridesOrDilationsAreOne:Ha,expandShapeToKeepDim:gr,exponent:Dc,exponents:Nc,fromStringArrayToUint8:Qr,fromUint8ToStringArray:Qt,getAxesPermutation:pr,getBroadcastDims:Zt,getComplexWithIndex:$c,getEinsumComputePath:Lc,getEinsumPermutation:Oc,getFusedBiasGradient:vl,getFusedDyActivation:xl,getImageCenter:dc,getInnerMostAxes:mr,getPermuted:pc,getRaggedRank:Ur,getReductionAxes:xn,getReshaped:gc,getReshapedPermuted:mc,getRowPartitionTypesHelper:Cr,getSliceBeginCoords:wc,getSliceSize:yc,getSparseFillEmptyRowsIndicesDenseShapeMismatch:Gr,getSparseFillEmptyRowsNegativeIndexErrorMessage:zr,getSparseFillEmptyRowsOutOfRangeIndexErrorMessage:qr,getSparseReshapeEmptyTensorZeroOutputDimErrorMessage:Hr,getSparseReshapeInputOutputMismatchErrorMessage:Jr,getSparseReshapeInputOutputMultipleErrorMessage:Kr,getSparseReshapeMultipleNegativeOneOutputDimErrorMessage:Vr,getSparseReshapeNegativeOutputDimErrorMessage:jr,getSparseSegmentReductionIndicesOutOfRangeErrorMessage:Yr,getSparseSegmentReductionNegativeSegmentIdsErrorMessage:ln,getSparseSegmentReductionNonIncreasingSegmentIdsErrorMessage:Xr,getSparseSegmentReductionSegmentIdOutOfRangeErrorMessage:Zr,getUndoAxesPermutation:il,isIdentityPermutation:Cc,log:ki,mergeRealAndImagArrays:an,prepareAndValidate:Ar,prepareSplitSize:Wc,segment_util:Vc,shouldFuse:Fl,slice_util:cc,splitRealAndImagArrays:vc,stridesOrDilationsArePositive:Ka,tupleValuesAreOne:sn,upcastType:Be,validateDefaultValueShape:Wr,validateInput:wr,validateUpdateShape:Mn,warn:It},Symbol.toStringTag,{value:"Module"}));/**
 * @license
 * Copyright 2017 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */Vl();/**
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
 */function Ut(e,t){Array.isArray(e)||(e=[e]),e.forEach(n=>{n!=null&&T(n.dtype!=="complex64",()=>`${t} does not support complex64 tensors in the CPU backend.`)})}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */function to(e){const t=new Float32Array(e.length);for(let n=0;n<e.length;++n)t[n]=Math.abs(e[n]);return t}const jc=e=>{const{x:t}=e.inputs,n=e.backend;Ut(t,"abs");let s=new Float32Array(F(t.shape));const r=n.data.get(t.dataId).values;return s=to(r),n.makeOutput(s,t.shape,t.dtype)},vg={kernelName:ws,backendName:"cpu",kernelFunc:jc};/**
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
 */function tt(e){return(t,n,s,r,o)=>{const i=we(t,n),a=i.length,l=Z(i),c=F(i),h=Tt(o,c),u=t.length,f=n.length,g=Z(t),m=Z(n),y=Zt(t,i),p=Zt(n,i);if(y.length+p.length===0)for(let w=0;w<h.length;++w)h[w]=e(s[w%s.length],r[w%r.length]);else for(let w=0;w<h.length;++w){const b=ge(w,a,l),I=b.slice(-u);y.forEach(B=>I[B]=0);const k=Ht(I,u,g),v=b.slice(-f);p.forEach(B=>v[B]=0);const $=Ht(v,f,m);h[w]=e(s[k],r[$])}return[h,i]}}/**
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
 */function Le(e){const{inputs:t,backend:n}=e,{real:s,imag:r}=t,o=n.data.get(s.dataId).values,i=n.data.get(r.dataId).values,a=n.makeTensorInfo(s.shape,"complex64"),l=n.data.get(a.dataId);return l.complexTensorInfos={real:n.makeTensorInfo(s.shape,"float32",o),imag:n.makeTensorInfo(r.shape,"float32",i)},a}const Mg={kernelName:Is,backendName:"cpu",kernelFunc:Le};/**
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
 */function cn(e,t,n="float32"){if(n==="complex64"){const r=cn(e,t,"float32"),o=cn(e,t,"float32");return Le({inputs:{real:r,imag:o},backend:e})}const s=pt(F(t),n);return e.makeTensorInfo(t,n,s)}/**
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
 */function un(e){const{inputs:t,backend:n}=e,{x:s}=t;return n.incRef(s.dataId),{dataId:s.dataId,shape:s.shape,dtype:s.dtype}}const Fg={kernelName:gn,backendName:"cpu",kernelFunc:un};/**
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
 */function eo(e){const{inputs:t,backend:n}=e,{input:s}=t,r=n.data.get(s.dataId).complexTensorInfos.real,o=n.data.get(r.dataId).values;return n.makeTensorInfo(r.shape,r.dtype,o)}const $g={kernelName:di,backendName:"cpu",kernelFunc:eo};/**
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
 */function no(e,t,n,s){if(s==="int32"){const r=Int32Array.from(e);return[t,"int32",r]}if(s==="bool"){const r=me([0],n),[o,i]=tt((a,l)=>a!==l?1:0)(t,[],e,r,"bool");return[i,"bool",o]}throw new Error(`Error in Cast: failed to cast ${n} to ${s}`)}function he(e){const{inputs:t,backend:n,attrs:s}=e,{x:r}=t,{dtype:o}=s;if(o==="complex64"){if(r.dtype==="complex64")return un({inputs:{x:r},backend:n});const h=cn(n,r.shape,r.dtype),u=he({inputs:{x:r},backend:n,attrs:{dtype:"float32"}}),f=Le({inputs:{real:u,imag:h},backend:n});return n.disposeIntermediateTensorInfo(h),n.disposeIntermediateTensorInfo(u),f}if(r.dtype==="complex64"){const h=eo({inputs:{input:r},backend:n}),u=he({inputs:{x:h},backend:n,attrs:{dtype:o}});return n.disposeIntermediateTensorInfo(h),u}if(!cs(r.dtype,o)){const h=un({inputs:{x:r},backend:n});return{dataId:h.dataId,shape:h.shape,dtype:o}}const i=n.data.get(r.dataId).values,[a,l,c]=no(i,r.shape,r.dtype,o);return n.makeTensorInfo(a,l,c)}const Rg={kernelName:fn,backendName:"cpu",kernelFunc:he};/**
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
 */function rt(e,t,n,s){return n==null?({inputs:r,backend:o})=>{const{a:i,b:a}=r,l=o;Ut([i,a],e);const c=l.data.get(i.dataId).values,h=l.data.get(a.dataId).values,u=i.dtype==="string"?Qt(c):c,f=i.dtype==="string"?Qt(h):h,g=s||i.dtype,[m,y]=t(i.shape,a.shape,u,f,g);return l.makeTensorInfo(y,g,m)}:({inputs:r,backend:o})=>{const{a:i,b:a}=r,l=o;if(i.dtype==="complex64"||a.dtype==="complex64"){const c=he({inputs:{x:i},backend:l,attrs:{dtype:"complex64"}}),h=l.data.get(c.dataId),u=h.complexTensorInfos.real,f=h.complexTensorInfos.imag,g=l.data.get(u.dataId).values,m=l.data.get(f.dataId).values,y=he({inputs:{x:a},backend:l,attrs:{dtype:"complex64"}}),p=l.data.get(y.dataId),w=p.complexTensorInfos.real,b=p.complexTensorInfos.imag,I=l.data.get(w.dataId).values,k=l.data.get(b.dataId).values,[v,$,B]=n(i.shape,a.shape,g,m,I,k),R=l.makeTensorInfo(B,"float32",v),E=l.makeTensorInfo(B,"float32",$),S=Le({inputs:{real:R,imag:E},backend:l});return l.disposeIntermediateTensorInfo(c),l.disposeIntermediateTensorInfo(y),l.disposeIntermediateTensorInfo(R),l.disposeIntermediateTensorInfo(E),S}else{const c=l.data.get(i.dataId).values,h=l.data.get(a.dataId).values,u=s||i.dtype,[f,g]=t(i.shape,a.shape,c,h,u);return l.makeTensorInfo(g,u,f)}}}function Nn(e){return(t,n,s,r,o,i)=>{const a=we(t,n),l=F(a),c=a.length,h=Z(a),u=Tt("float32",l),f=Tt("float32",l),g=Zt(t,a),m=Zt(n,a),y=an(s,r),p=an(o,i),w=t.length,b=Z(t),I=n.length,k=Z(n);if(g.length+m.length===0)for(let v=0;v<u.length;v++){const $=v%y.length,B=v%p.length,R=e(y[$*2],y[$*2+1],p[B*2],p[B*2+1]);u[v]=R.real,f[v]=R.imag}else for(let v=0;v<u.length;v++){const $=ge(v,c,h),B=$.slice(-w);g.forEach(A=>B[A]=0);const R=Ht(B,w,b),E=$.slice(-I);m.forEach(A=>E[A]=0);const S=Ht(E,I,k),d=e(y[R*2],y[R*2+1],p[S*2],p[S*2+1]);u[v]=d.real,f[v]=d.imag}return[u,f,a]}}/**
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
 */const so=tt(((e,t)=>e+t)),Hc=Nn(((e,t,n,s)=>({real:e+n,imag:t+s}))),Kc=rt(_e,so,Hc),Ng={kernelName:_e,backendName:"cpu",kernelFunc:Kc};/**
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
 */function Jc(e,t,n,s,r){const o=F(s),i=pt(r,n);for(let a=0;a<e.length;a++){const l=e[a];if(l<0)throw new Error("Input x must be non-negative!");l>=r||(o>0?i[l]+=t[a]:i[l]+=1)}return i}function Xc(e,t,n,s=!1){const r=e.shape[0],o=e.shape[1],i=st([r,n],t.dtype);for(let a=0;a<r;a++)for(let l=0;l<o;l++){const c=e.get(a,l);if(c<0)throw new Error("Input x must be non-negative!");c>=n||(s?i.set(1,a,c):t.size>0?i.set(i.get(a,c)+t.get(a,l),a,c):i.set(i.get(a,c)+1,a,c))}return i}/**
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
 */const ro=tt(((e,t)=>e&t)),Zc=rt(ys,ro),Dg={kernelName:ys,backendName:"cpu",kernelFunc:Zc};/**
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
 */function wt(e){return(t,n,s)=>{const r=j(n,t.length);for(let o=0;o<t.length;++o)r[o]=e(t[o],s);return r}}/**
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
 */function oo(e,t,n){const s=wt(t);return Mt(e,s,n)}function Mt(e,t,n){return({inputs:s,attrs:r,backend:o})=>{const{x:i}=s;Ut(i,e);const a=o,l=a.data.get(i.dataId).values;let c;if(i.dtype==="string"){if(!Array.isArray(l))throw new Error("String tensor's value was not an instance of Array");c=Qt(l)}else c=l;const h=n||i.dtype,u=t(c,h,r);return a.makeTensorInfo(i.shape,h,u)}}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const io=wt(e=>Math.ceil(e)),Yc=Mt(bs,io),_g={kernelName:bs,backendName:"cpu",kernelFunc:Yc};/**
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
 */function Qc(e,t,n,s){const r=j(n,F(t));if(s&&n!=="string"){let o=0;e.forEach(i=>{const a=F(i.shape);r.set(i.vals,o),o+=a})}else{let o=0;e.forEach(i=>{const a=n==="string"?Qt(i.vals):i.vals;let l=0;for(let c=0;c<i.shape[0];++c){const h=c*t[1]+o;for(let u=0;u<i.shape[1];++u)r[h+u]=a[l++]}o+=i.shape[1]})}return r}/**
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
 */const ao=tt((e,t)=>e===t?1:0),tu=rt(Ss,ao,null,"bool"),Bg={kernelName:Ss,backendName:"cpu",kernelFunc:tu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const lo=wt(e=>Math.exp(e)),eu=Mt(Es,lo,"float32"),Og={kernelName:Es,backendName:"cpu",kernelFunc:eu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const co=wt(e=>Math.expm1(e)),nu=Mt(ks,co),Pg={kernelName:ks,backendName:"cpu",kernelFunc:nu};/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const uo=wt(e=>Math.floor(e)),su=Mt(Ts,uo),Lg={kernelName:Ts,backendName:"cpu",kernelFunc:su};/**
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
 */const ho=tt((e,t)=>Math.floor(e/t)),ru=rt(dn,ho,null,"int32"),Cg={kernelName:dn,backendName:"cpu",kernelFunc:ru};/**
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
 */function ou(e,t,n,s,r,o,i,a,l){const c=st([s,o],n);for(let h=0;h<s;h++){const u=[];let f=0;for(let g=0;g<r;g++){const m=e[h*r+g];f+=m*i[g],u.push(m)}if(f<0||f>=l/o)throw new Error(`Invalid indices: ${u} does not index into ${a}`);for(let g=0;g<o;g++)c.values[h*o+g]=t.get(...t.indexToLoc(f*o+g))}return c}/**
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
 */function iu(e,t,n){const s=st(n,e.dtype);for(let r=0;r<s.size;++r){const i=s.indexToLoc(r).slice(),a=i[0],l=i[2],c=t.locToIndex([a,l]);i[2]=t.values[c];const h=e.locToIndex(i);0<=h&&h<e.values.length&&(s.values[r]=e.values[h])}return s}/**
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
 */const fo=tt((e,t)=>e>t?1:0),au=rt(As,fo,null,"bool"),Ug={kernelName:As,backendName:"cpu",kernelFunc:au};/**
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
 */const go=tt((e,t)=>e>=t?1:0),lu=rt(xs,go,null,"bool"),Wg={kernelName:xs,backendName:"cpu",kernelFunc:lu};/**
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
 */const po=tt((e,t)=>e<t?1:0),cu=rt(vs,po,null,"bool"),Gg={kernelName:vs,backendName:"cpu",kernelFunc:cu};/**
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
 */const mo=tt((e,t)=>e<=t?1:0),uu=rt(Ms,mo,null,"bool"),zg={kernelName:Ms,backendName:"cpu",kernelFunc:uu};/**
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
 */function hu(e,t,n){const s=(t-e)/(n-1),r=pt(n,"float32");r[0]=e;for(let o=1;o<r.length;o++)r[o]=r[o-1]+s;return r}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const wo=wt(e=>Math.log(e)),fu=Mt(Fs,wo),qg={kernelName:Fs,backendName:"cpu",kernelFunc:fu};/**
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
 */function du(e,t,n,s){const r=Tt(s,F(n));for(let o=0;o<r.length;++o){const i=o*t;let a=e[i];for(let l=0;l<t;++l){const c=e[i+l];(Number.isNaN(c)||c>a)&&(a=c)}r[o]=a}return r}/**
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
 */const yo=tt(((e,t)=>Math.max(e,t))),gu=rt(pn,yo),Vg={kernelName:pn,backendName:"cpu",kernelFunc:gu};/**
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
 */const bo=tt(((e,t)=>Math.min(e,t))),pu=rt($s,bo),jg={kernelName:$s,backendName:"cpu",kernelFunc:pu};/**
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
 */const Dn=tt(((e,t)=>e*t)),mu=Nn(((e,t,n,s)=>({real:e*n-t*s,imag:e*s+t*n}))),wu=rt(mn,Dn,mu),Hg={kernelName:mn,backendName:"cpu",kernelFunc:wu};/**
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
 */function Io(e,t,n){const s=Ws(-1,n);return Dn([],t,s,e,n)}function yu(e){const{inputs:t,backend:n}=e,{x:s}=t;Ut(s,"neg");const r=n.data.get(s.dataId).values,[o,i]=Io(r,s.shape,s.dtype);return n.makeTensorInfo(i,s.dtype,o)}const Kg={kernelName:ci,backendName:"cpu",kernelFunc:yu};/**
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
 */const So=tt(((e,t)=>e!==t?1:0)),bu=rt(Rs,So,null,"bool"),Jg={kernelName:Rs,backendName:"cpu",kernelFunc:bu};/**
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
 */function Eo(e,t,n,s,r){const o=t.length,i=F(t),a=Z(t),l=Z(r),c=Tt(n,F(r));for(let h=0;h<i;++h){const u=ge(h,o,a),f=new Array(u.length);for(let m=0;m<f.length;m++)f[m]=u[s[m]];const g=Ht(f,o,l);c[g]=e[h]}return c}/**
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
 */function ko(e){const{inputs:t,attrs:n,backend:s}=e,{x:r}=t,{perm:o}=n;Ut(r,"transpose");const i=r.shape.length,a=new Array(i);for(let u=0;u<a.length;u++)a[u]=r.shape[o[u]];const l=s.data.get(r.dataId).values,c=Eo(l,r.shape,r.dtype,o,a);return{dataId:s.write(c,a,r.dtype),shape:a,dtype:r.dtype}}const Xg={kernelName:Ii,backendName:"cpu",kernelFunc:ko};/**
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
 */function To(e,t,n,s){const[r,o]=dr(e,s),i=Be(t,"int32"),a=pt(F(r),i),l=F(o);for(let c=0;c<a.length;++c){const h=c*l;let u=1;for(let f=0;f<l;++f)u*=n[h+f];a[c]=u}return{outVals:a,outShape:r,outDtype:i}}function Iu(e){const{inputs:t,backend:n,attrs:s}=e,{x:r}=t,{axis:o,keepDims:i}=s;Ut(r,"prod");const a=r.shape.length,l=Re(o,r.shape),c=pr(l,a);let h=l,u=r;const f=[];c!=null&&(u=ko({inputs:{x:r},backend:n,attrs:{perm:c}}),f.push(u),h=mr(h.length,a));const g=n.data.get(u.dataId).values,{outVals:m,outShape:y,outDtype:p}=To(u.shape,u.dtype,g,h);let w=y;return i&&(w=gr(y,l)),f.forEach(b=>n.disposeIntermediateTensorInfo(b)),n.makeTensorInfo(w,p,m)}const Zg={kernelName:fi,backendName:"cpu",kernelFunc:Iu};/**
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
 */function Su(e,t,n){e.forEach((s,r)=>{if(s<0||s>=n){const o=ge(r,t.length,Z(t)).join(",");throw new Error(`indices[${o}] = ${s} is not in [0, ${n})`)}})}function Eu(e,t){for(let n=0;n<e.length;++n){const s=e[n],r=n===e.length-1?t:e[n+1].length;if(s.length===0)throw new Error("Ragged splits may not be empty");if(s[0]<0)throw new Error("Ragged splits must be non-negative");if(s[s.length-1]>r)throw new Error("Ragged splits must not point past values");for(let o=1;o<s.length;++o)if(s[o-1]>s[o])throw new Error("Ragged splits must be sorted in ascending order")}}function ku(e,t,n,s){const r=[];let o=0;const i=t.length-1+n.length,a=new Array(i).fill(null).map(()=>[0]);Eu(n,s);let l=1;for(let c=0;c<t.length-1;++c){l*=t[c];const h=t[c+1];for(let u=1;u<l+1;++u)a[c].push(u*h)}for(let c=0;c<e.length;++c){let h=e[c],u=e[c]+1;for(let f=0;f<n.length;++f){const g=n[f],m=f+t.length-1;if(m>=0){const y=a[m],p=y[y.length-1]-g[h];for(let w=h;w<u;++w)a[m].push(g[w+1]+p)}h=g[h],u=g[u]}u!==h&&(r.push([h,u]),o+=u-h)}return{outSplits:a,valueSlices:r,numValues:o}}function Tu(e){const t=[];for(let n=0;n<e.length;++n){const s=e[n].length,r=j("int32",s);t.push(r),e[n].forEach((o,i)=>r[i]=o)}return t}function ns(e,t){const n=e.slice(0,t);for(;n.length<t;)n.push(1);for(let s=t;s<e.length;s++)n[t-1]*=e[s];return n}function Au(e,t,n,s,r,o){const i=ns(t,2)[1],a=ns(o,2)[1];let l=0;for(const c of n)for(let h=c[0];h<c[1];++h){for(let u=0;u<s;++u)r[l*a+u]=e[h*i+u];++l}}function xu(e,t,n,s,r){const o=t.slice();o[0]=r;const i=j(n,F(o)),a=e.length,l=a===0?0:a/t[0];return Au(e,t,s,l,i,o),[i,o]}function vu(e,t,n,s,r,o,i,a){if(e.length===0)throw new Error("paramsNestedSplits must be non empty");if(t[0].length===0)throw new Error("Split tensors must not be scalars");const l=t[0][0]-1;if(Su(o,i,l),s.length===0)throw new Error("params.rank must be nonzero");const c=s[0],{outSplits:h,valueSlices:u,numValues:f}=ku(o,i,e,c),g=Tu(h),m=xu(n,s,r,u,f);return[g,m[0],m[1]]}/**
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
 */const ss=2147483647;function Mu(e,t,n,s,r,o,i){if(t.length>1)throw new Error("starts must be a scalar or vector");if(r.length>1)throw new Error("limits must be a scalar or vector");if(i.length>1)throw new Error("deltas must be a scalar or vector");const a=t.length===0,l=r.length===0,c=i.length===0,h=[];a||h.push(t[0]),l||h.push(r[0]),c||h.push(i[0]);for(let p=1;p<h.length;++p)if(h[p]!==h[p-1])throw new Error("starts, limits, and deltas must have the same shape");const u=h.length===0?1:h[0],f=j("int32",u+1);f[0]=0;for(let p=0;p<u;++p){const w=a?e[0]:e[p],b=l?s[0]:s[p],I=c?o[0]:o[p];if(I===0)throw new Error("Requires delta != 0");let k;if(I>0&&b<w||I<0&&b>w)k=0;else if(k=Math.ceil(Math.abs((b-w)/I)),k>ss)throw new Error(`Requires ((limit - start) / delta) <= ${ss}`);f[p+1]=f[p]+k}const g=f[u],m=j(n,g);let y=0;for(let p=0;p<u;++p){const w=f[p+1]-f[p];let b=a?e[0]:e[p];const I=c?o[0]:o[p];for(let k=0;k<w;++k)m[y++]=b,b+=I}return[f,m]}/**
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
 */var ut=ft;class Fe{constructor(t,n,s,r,o,i,a,l,c,h){this.shape=t,this.shapeShape=n,this.values=s,this.valuesShape=r,this.valuesDType=o,this.defaultValue=i,this.defaultValueShape=a,this.rowPartitionValues=l,this.rowPartitionValuesShapes=c,this.rowPartitionTypes=Cr(h),this.raggedRank=Ur(this.rowPartitionTypes)}getRowPartitionTypeByDimension(t){return this.rowPartitionTypes[0]===ut.FIRST_DIM_SIZE?this.rowPartitionTypes[t+1]:this.rowPartitionTypes[t]}getRowPartitionTensor(t){return this.rowPartitionTypes[0]===ut.FIRST_DIM_SIZE?this.rowPartitionValues[t+1]:this.rowPartitionValues[t]}getMaxWidth(t){const n=this.getRowPartitionTensor(t-1);switch(this.getRowPartitionTypeByDimension(t-1)){case ut.VALUE_ROWIDS:return Fe.getMaxWidthValueRowID(n);case ut.ROW_SPLITS:return Fe.getMaxWidthRowSplit(n);default:throw new Error(`Cannot handle partition type ${ut[this.getRowPartitionTypeByDimension(t-1)]}`)}}static getMaxWidthRowSplit(t){const n=t.length;if(n===0||n===1)return 0;let s=0;for(let r=0;r<n-1;++r){const o=t[r+1]-t[r];o>s&&(s=o)}return s}static getMaxWidthValueRowID(t){const n=t.length;if(n===0)return 0;let s=0,r=t[0],o=0;for(let i=1;i<n;++i){const a=t[i];a!==r&&(r=a,o=Math.max(i-s,o),s=i)}return Math.max(n-s,o)}tensorShapeFromTensor(t,n,s=!0){if(n.length===0){if(t[0]===-1)return[];throw new Error("The only valid scalar shape tensor is the fully unknown shape specified as -1.")}return os(t,s)}calculateOutputSize(t){const n=this.valuesShape,s=this.defaultValueShape;Wr(s,n);const r=this.tensorShapeFromTensor(this.shape,this.shapeShape),i=Lr(this.raggedRank,r,n);i[0]<0&&(i[0]=t);for(let a=1;a<=this.raggedRank;++a)i[a]<0&&(i[a]=this.getMaxWidth(a));return i}calculateFirstParentOutputIndex(t,n,s){const r=Math.min(t,s),o=[];let i=0;for(let a=0;a<r;++a,i+=n)o.push(i);for(let a=r;a<t;++a)o.push(-1);return T(o.length===t,()=>"Final length of result must be equal to firstDimension."),o}calculateOutputIndexRowSplit(t,n,s,r){const o=t.length,i=[];for(let a=0;a<o-1;++a){const l=t[a+1]-t[a];let c=Math.min(r,l),h=n[a];h===-1&&(c=0);for(let u=0;u<c;++u)i.push(h),h+=s;for(let u=0;u<l-c;++u)i.push(-1)}if(o>0&&i.length!==t[o-1])throw new Error("Invalid row split size.");return i}calculateOutputIndexValueRowID(t,n,s,r){const o=t.length,i=[];if(o===0)return[];let a=0,l=t[0];if(l>=n.length)throw new Error(`Got currentValueRowId=${l}, which is not less than ${n.length}`);let c=n[l];i.push(c);for(let h=1;h<o;++h){const u=t[h];if(u===l)c>=0&&(++a,a<r?c+=s:c=-1);else{if(a=0,l=u,u>=n.length)throw new Error(`Got nextValueRowId=${u} which is not less than ${n.length}`);c=n[u]}i.push(c)}if(i.length!==t.length)throw new Error("Invalid row ids.");return i}calculateOutputIndex(t,n,s,r){const o=this.getRowPartitionTensor(t),i=this.getRowPartitionTypeByDimension(t);switch(i){case ut.VALUE_ROWIDS:return this.calculateOutputIndexValueRowID(o,n,s,r);case ut.ROW_SPLITS:if(o.length-1>n.length)throw new Error(`Row partition size is greater than output size: ${o.length-1} > ${n.length}`);return this.calculateOutputIndexRowSplit(o,n,s,r);default:throw new Error(`Unsupported partition type: ${ut[i]}`)}}getFirstDimensionSize(){const t=this.rowPartitionValues[0];if(this.rowPartitionTypes.length===0)throw new Error("No row_partition_types given.");const n=this.rowPartitionTypes[0];switch(n){case ut.FIRST_DIM_SIZE:return t[0];case ut.VALUE_ROWIDS:throw new Error("Cannot handle VALUE_ROWIDS in first dimension.");case ut.ROW_SPLITS:return this.rowPartitionValuesShapes[0][0]-1;default:throw new Error(`Cannot handle type ${ut[n]}`)}}compute(){if(this.rowPartitionValues[0].length<=0)throw new Error("Invalid first partition input. Tensor requires at least one element.");const n=this.getFirstDimensionSize(),s=this.calculateOutputSize(n),r=new Array(this.raggedRank+1);r[r.length-1]=1;for(let l=r.length-2;l>=0;--l)r[l]=r[l+1]*s[l+1];const o=os(s,!1),i=j(this.valuesDType,F(o));if(r[0]*s[0]>0){let l=this.calculateFirstParentOutputIndex(n,r[0],s[0]);for(let c=1;c<=this.raggedRank;++c)l=this.calculateOutputIndex(c-1,l,r[c],s[c]);this.setOutput(this.raggedRank,l,i,o)}return[o,i]}setOutput(t,n,s,r){if(s.length===0)return;const o=this.values,i=s;let a=r.slice();a=a.slice(t+1);const l=F(a),c=n.length;let h=this.defaultValue;if(h.length!==l&&h.length!==1){const m=this.defaultValueShape;z(()=>{const y=An(h,m);h=tl(y,a).dataSync()})}let u=0,f=0,g=0;for(let m=0;m<=c;++m){let y=m<c?n[m]:-1;if(y===g){++g;continue}if(f<g){const p=o.subarray(u*l),w=i.subarray(f*l),b=(g-f)*l;rs(w,p,b)}if(m>=c){const p=s.length;y=Math.floor(p/l)}if(y>g)if(this.defaultValue.length===1)i.subarray(g*l,y*l).fill(this.defaultValue[0]),g=y;else for(;y>g;){const p=i.slice(g*l);rs(p,h,l),++g}y<0?(u=m+1,f=g):(u=m,f=g,g=f+1)}}}function rs(e,t,n){for(let s=0;s<n;s++)e[s]=t[s]}function os(e,t){const n=[];for(let s of e){if(s<0){if(!t)throw new Error(`Dimension ${s} must be >= 0`);if(s<-1)throw new Error(`Dimension ${s} must be >= -1`);s=-1}n.push(s)}return n}function Fu(e,t,n,s,r,o,i,a,l,c){return new Fe(e,t,n,s,r,o,i,a,l,c).compute()}/**
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
 */function $u(e,t,n,s){const r=e===t,o=e<t&&n<0,i=t<e&&n>1;if(r||o||i)return pt(0,s);const a=Math.abs(Math.ceil((t-e)/n)),l=pt(a,s);t<e&&n===1&&(n=-1),l[0]=e;for(let c=1;c<l.length;c++)l[c]=l[c-1]+n;return l}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Ao=wt(e=>1/Math.sqrt(e)),Ru=Mt(Ns,Ao),Yg={kernelName:Ns,backendName:"cpu",kernelFunc:Ru};/**
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
 */function Nu(e,t,n,s,r,o,i,a,l,c){const h=[s/r,r],u=e.values,f=t.values;if(s===0)return st(n,t.dtype);const g=l instanceof xe?l:st(h,t.dtype);typeof l=="string"||typeof l=="number"?g.values.fill(l):typeof l=="boolean"&&g.values.fill(+l);for(let m=0;m<o;m++){const y=[];let p=0;for(let w=0;w<i;w++){const b=u[m*i+w];y.push(b),p+=b*a[w]}if(p<0||p>=s/r)throw new Error(`Invalid indices: ${y} does not index into ${n}`);for(let w=0;w<r;w++)c?g.values[p*r+w]+=f[m*r+w]:g.values[p*r+w]=t.rank===0?f[0]:f[m*r+w]}return g}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Du=wt(e=>1/(1+Math.exp(-e))),_u=oo(wn,e=>1/(1+Math.exp(-e))),Qg={kernelName:wn,backendName:"cpu",kernelFunc:_u};/**
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
 */function xo(e,t,n,s,r){const o=Br(s,t,n),i=F(n),a=Z(s);if(o){const u=Or(t,a);return r==="string"?e.slice(u,u+i):e.subarray(u,u+i)}const l=r==="string"?Qt(e):e,c=st(s,r,l),h=st(n,r);for(let u=0;u<h.size;++u){const f=h.indexToLoc(u),g=f.map((m,y)=>m+t[y]);h.set(c.get(...g),...f)}return r==="string"?Qr(h.values):h.values}function Bu(e){const{inputs:t,backend:n,attrs:s}=e,{x:r}=t,{begin:o,size:i}=s;Ut(r,"slice");const[a,l]=Pr(r,o,i);xr(r,a,l);const c=n.data.get(r.dataId).values,h=xo(c,a,l,r.shape,r.dtype);return n.makeTensorInfo(l,r.dtype,h)}const tp={kernelName:wi,backendName:"cpu",kernelFunc:Bu};/**
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
 */function Ou(e,t,n,s,r,o,i){const a=t[0],l=o[0],c=new Array(l),h=new Array(a),u=t[1];if(l===0){if(a!==0)throw new Error(Gr(a));const p=j(n,0),w=j(r,0);return[p,[0,u],w,c,h]}let f=!0,g=0;const m=new Array(l).fill(0);for(let p=0;p<a;++p){const w=e[p*u];if(w<0)throw new Error(zr(p,w));if(w>=l)throw new Error(qr(p,w,l));++m[w],f=f&&w>=g,g=w}let y=!0;for(let p=0;p<l;++p){const w=m[p]===0;c[p]=w,y=y&&!w,m[p]=Math.max(m[p],1),p>0&&(m[p]+=m[p-1])}if(y&&f){const p=e,w=s;for(let b=0;b<a;++b)h[b]=b;return[p,[a,u],w,c,h]}else{const p=m[l-1],w=j(n,p*u),b=j(r,p),I=new Array(l).fill(0);for(let k=0;k<a;++k){const v=e[k*u],$=I[v],B=(v===0?0:m[v-1])+$;I[v]++;for(let R=0;R<u;++R)w[B*u+R]=e[k*u+R];b[B]=s[k],h[k]=B}for(let k=0;k<l;++k)if(I[k]===0){const $=k===0?0:m[k-1];w[$*u+0]=k;for(let B=1;B<u;++B)w[$*u+B]=0;b[$]=i}return[w,[p,u],b,c,h]}}/**
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
 */function Pu(e,t,n,s,r){const o=F(s),i=t[0],a=r.length,l=[];let c=1,h=-1;for(let p=0;p<a;++p){const w=r[p];if(w===-1){if(h!==-1)throw new Error(Vr(h,p));h=p,l.push(1)}else{if(w<0)throw new Error(jr(p,w));c*=w,l.push(w)}}if(h!==-1){if(c<=0)throw new Error(Hr());const p=Math.trunc(o/c);if(c*p!==o)throw new Error(Kr(s,l));l[h]=p}if(F(l)!==o)throw new Error(Jr(s,l));const f=s.length,g=[];if(f>0){g[f-1]=1;for(let p=f-2;p>=0;--p)g[p]=g[p+1]*s[p+1]}const m=[];if(a>0){m[a-1]=1;for(let p=a-2;p>=0;--p)m[p]=m[p+1]*l[p+1]}const y=j(n,i*a);for(let p=0;p<i;++p){let w=0;for(let b=0;b<f;++b)w+=e[p*f+b]*g[b];for(let b=0;b<a;++b)y[p*a+b]=Math.trunc(w/m[b]),w%=m[b]}return[y,[i,a],l]}/**
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
 */function Lu(e,t,n,s,r,o=!1,i=0){const a=s.length,l=[t[0],e.length/t[0]],c=l[1],u=a>0?r[a-1]+1:0;if(u<0)throw new Error(ln());const f=t.slice();f[0]=u;const g=f.reduce((I,k)=>I*k,1),m=j(n,g);if(a===0)return u>0&&m.fill(i),[m,f];if(u<=0)throw new Error(ln());let y=0,p=1,w=0,b=r[y];for(;;){let I=0;if(p<a){if(I=r[p],b===I){++p;continue}if(b>=I)throw new Error(Xr())}if(b<0||b>=u)throw new Error(Zr(b,u));b>w&&m.fill(i,w*c,b*c);for(let k=y;k<p;++k){const v=s[k];if(v<0||v>=l[0])throw new Error(Yr(k,s[k],l[0]));for(let $=0;$<c;$++)m[b*c+$]+=e[v*c+$]}if(o)for(let k=0;k<c;k++)m[b*c+k]/=p-y;if(y=p,++p,w=b+1,b=I,p>a)break}return w<u&&m.fill(i,w*c,u*c),[m,f]}/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the License);
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */const Cu=wt(e=>Math.sqrt(e)),Uu=oo(yn,e=>Math.sqrt(e)),ep={kernelName:yn,backendName:"cpu",kernelFunc:Uu};/**
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
 */const vo=tt(((e,t)=>{const n=e-t;return n*n})),Wu=rt(Ds,vo),np={kernelName:Ds,backendName:"cpu",kernelFunc:Wu};/**
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
 */const Mo=wt((e,t)=>{const{pattern:n,replaceGlobal:s,rewrite:r}=t;return e.replace(new RegExp(n,s?"g":""),r)}),Gu=Mt(_s,Mo),sp={kernelName:_s,backendName:"cpu",kernelFunc:Gu};/**
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
 */function zu(e,t,n,s){const r=st(e,t.dtype);for(let o=0;o<r.size;o++){const i=r.indexToLoc(o),a=new Array(i.length);for(let l=0;l<a.length;l++)a[l]=i[l]*n[l]+s[l];r.set(t.get(...a),...i)}return r}/**
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
 */class qu{constructor(t,n,s,r,o,i){this.separator=Nt(t),this.nGramWidths=n,this.leftPad=Nt(s),this.rightPad=Nt(r),this.padWidth=o,this.preserveShort=i}getPadWidth(t){return Math.min(this.padWidth<0?t-1:this.padWidth,t-1)}getNumNGrams(t,n){const s=this.getPadWidth(n);return Math.max(0,t+2*s-n+1)}createNGrams(t,n,s,r,o,i){for(let a=0;a<o;++a){const l=this.getPadWidth(i),c=Math.max(0,l-a),h=Math.max(0,l-(o-(a+1))),u=i-(c+h),f=n+(c>0?0:a-l);let g=0;g+=c*this.leftPad.length;for(let b=0;b<u;++b)g+=t[f+b].length;g+=h*this.rightPad.length;const m=c+h+u-1;g+=m*this.separator.length,s[r+a]=new Uint8Array(g);const y=s[r+a];let p=0;const w=b=>b.forEach(I=>y[p++]=I);for(let b=0;b<c;++b)w(this.leftPad),w(this.separator);for(let b=0;b<u-1;++b)w(t[f+b]),w(this.separator);if(u>0){w(t[f+u-1]);for(let b=0;b<h;++b)w(this.separator),w(this.rightPad)}else{for(let b=0;b<h-1;++b)w(this.rightPad),w(this.separator);w(this.rightPad)}}}compute(t,n){const s=t.length,r=n.length;if(r>0){let l=n[0];if(l!==0)throw new Error(`First split value must be 0, got ${l}`);for(let c=1;c<r;++c){let h=n[c]>=l;if(h=h&&n[c]<=s,!h)throw new Error(`Invalid split value ${n[c]}, must be in [${l}, ${s}]`);l=n[c]}if(l!==s)throw new Error(`Last split value must be data size. Expected ${s}, got ${l}`)}const o=r-1,i=j("int32",r);if(s===0||r===0){const l=new Array(s);for(let c=0;c<=o;++c)i[c]=0;return[l,i]}i[0]=0;for(let l=1;l<=o;++l){const c=n[l]-n[l-1];let h=0;this.nGramWidths.forEach(u=>{h+=this.getNumNGrams(c,u)}),this.preserveShort&&c>0&&h===0&&(h=1),i[l]=i[l-1]+h}const a=new Array(i[o]);for(let l=0;l<o;++l){const c=n[l];let h=i[l];if(this.nGramWidths.forEach(u=>{const f=n[l+1]-n[l],g=this.getNumNGrams(f,u);this.createNGrams(t,c,a,h,g,u),h+=g}),this.preserveShort&&h===i[l]){const u=n[l+1]-n[l];if(u===0)continue;const f=u+2*this.padWidth;this.createNGrams(t,c,a,h,1,f)}}return[a,i]}}function Vu(e,t,n,s,r,o,i,a){return new qu(n,s,r,o,i,a).compute(e,t)}/**
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
 */function ju(e,t,n,s){if(!e.length)return;if(t.length===0){for(let o=0;o<e.length;++o)s.push(e.subarray(o,o+1));return}if(t.length===1){const o=t[0];let i=e.indexOf(o);for(;i!==-1;){const a=e.subarray(0,i);(!n||a.length!==0)&&s.push(a),e=e.subarray(i+1),i=e.indexOf(o)}(!n||e.length!==0)&&s.push(e);return}let r=0;for(let o=0;o<e.length+1;o++)if(o===e.length||t.indexOf(e[o])!==-1){const i=e.subarray(r,o);(!n||i.length!==0)&&s.push(i),r=o+1}}function Hu(e,t,n){const s=e.length,r=[];let o=0,i=0;const a=new Array(s);for(let f=0;f<s;++f){const g=r.length;ju(e[f],t,n,r);const m=r.length-g;a[f]=m,o+=m,i=Math.max(i,m)}const l=j("int32",o*2),c=new Array(o),h=[s,i];let u=0;for(let f=0;f<s;++f)for(let g=0;g<a[f];++g)l[u*2]=f,l[u*2+1]=g,c[u]=r[u],++u;return[l,c,h]}/**
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
 */function Ku(e,t){const n=j("int32",e.length);for(let s=0;s<e.length;++s)n[s]=Us(e[s]).modulo(t).getLowBitsUnsigned();return n}/**
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
 */const Fo=tt(((e,t)=>e-t)),Ju=Nn(((e,t,n,s)=>({real:e-n,imag:t-s}))),Xu=rt(bn,Fo,Ju),rp={kernelName:bn,backendName:"cpu",kernelFunc:Xu};/**
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
 */function Zu(e,t){const n=new Array(e.rank);for(let r=0;r<n.length;r++)n[r]=e.shape[r]*t[r];const s=st(n,e.dtype);for(let r=0;r<s.values.length;++r){const o=s.indexToLoc(r),i=new Array(e.rank);for(let l=0;l<i.length;l++)i[l]=o[l]%e.shape[l];const a=e.locToIndex(i);s.values[r]=e.values[a]}return s}/**
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
 */const re=(e,t)=>{const n=t.value-e.value;return n===0?e.index-t.index:n};function $o(e,t,n=0,s=e.length-1){for(;s>n;){if(s-n>600){const a=s-n+1,l=t-n+1,c=Math.log(a),h=.5*Math.exp(2*c/3),u=.5*Math.sqrt(c*h*(a-h)/a)*Math.sign(l-a/2),f=Math.max(n,Math.floor(t-l*h/a+u)),g=Math.min(s,Math.floor(t+(a-l)*h/a+u));$o(e,t,f,g)}const r=e[t];let o=n,i=s;for(yt(e,n,t),re(e[s],r)>0&&yt(e,n,s);o<i;){for(yt(e,o,i),o++,i--;re(e[o],r)<0;)o=o+1;for(;re(e[i],r)>0;)i=i-1}re(e[n],r)===0?yt(e,n,i):(i=i+1,yt(e,i,s)),i<=t&&(n=i+1),t<=i&&(s=i-1)}}function Yu(e,t,n,s,r){const o=t[t.length-1],[i,a]=[e.length/o,o],l=Tt(n,i*s),c=Tt("int32",i*s);for(let u=0;u<i;u++){const f=u*a,g=e.subarray(f,f+a);let m=new Array(g.length);g.forEach((b,I)=>m[I]={value:b,index:I}),s<m.length&&($o(m,s),m=m.slice(0,s)),r&&m.sort(re);const y=u*s,p=l.subarray(y,y+s),w=c.subarray(y,y+s);for(let b=0;b<s;b++)p[b]=m[b].value,w[b]=m[b].index}const h=t.slice();return h[h.length-1]=s,[st(h,n,l),st(h,"int32",c)]}/**
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
 */function Qu(e,t,n,s){const r=Re(t,n)[0],o=[1,n[0],1];for(let m=0;m<r;m++)o[0]*=n[m];o[1]=n[r];for(let m=r+1;m<n.length;m++)o[2]*=n[m];const i=new Map,a=new Int32Array(n[r]),l=new xe(o,s,e),c=[],h=o[0]===1&&o[2]===1;for(let m=0;m<n[r];m++){let y;if(h)y=e[m].toString();else{const w=[];for(let b=0;b<o[0];b++)for(let I=0;I<o[2];I++)w.push(l.get(b,m,I));y=w.join(",")}const p=i.get(y);if(p!=null)a[m]=p;else{const w=i.size;i.set(y,w),a[m]=w,c.push(m)}}const u=o.slice();u[1]=i.size;const f=new xe(u,s);c.forEach((m,y)=>{for(let p=0;p<o[0];p++)for(let w=0;w<o[2];w++)f.set(l.get(p,m,w),p,y,w)});const g=n.slice();return g[r]=u[1],{outputValues:f.values,outputShape:g,indices:a}}/**
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
 */const op=Object.freeze(Object.defineProperty({__proto__:null,addImpl:so,bincountImpl:Jc,bincountReduceImpl:Xc,bitwiseAndImpl:ro,castImpl:no,ceilImpl:io,concatImpl:Qc,equalImpl:ao,expImpl:lo,expm1Impl:co,floorDivImpl:ho,floorImpl:uo,gatherNdImpl:ou,gatherV2Impl:iu,greaterEqualImpl:go,greaterImpl:fo,lessEqualImpl:mo,lessImpl:po,linSpaceImpl:hu,logImpl:wo,maxImpl:du,maximumImpl:yo,minimumImpl:bo,multiplyImpl:Dn,negImpl:Io,notEqualImpl:So,prodImpl:To,raggedGatherImpl:vu,raggedRangeImpl:Mu,raggedTensorToTensorImpl:Fu,rangeImpl:$u,rsqrtImpl:Ao,scatterImpl:Nu,sigmoidImpl:Du,simpleAbsImpl:to,sliceImpl:xo,sparseFillEmptyRowsImpl:Ou,sparseReshapeImpl:Pu,sparseSegmentReductionImpl:Lu,sqrtImpl:Cu,squaredDifferenceImpl:vo,staticRegexReplaceImpl:Mo,stridedSliceImpl:zu,stringNGramsImpl:Vu,stringSplitImpl:Hu,stringToHashBucketFastImpl:Ku,subImpl:Fo,tileImpl:Zu,topKImpl:Yu,transposeImpl:Eo,uniqueImpl:Qu},Symbol.toStringTag,{value:"Module"}));export{Gh as $,ws as A,yh as B,fn as C,bs as D,kh as E,Is as F,ri as G,mt as H,Th as I,Ah as J,xh as K,vh as L,Mh as M,Fh as N,$h as O,Rh as P,Nh as Q,Bh as R,Dh as S,_h as T,eh as U,Oh as V,Ph as W,Lh as X,Ch as Y,Uh as Z,Wh as _,nh as a,Af as a$,qh as a0,zh as a1,Vh as a2,x as a3,ps as a4,Ec as a5,kc as a6,Tc as a7,Ac as a8,xc as a9,sf as aA,rf as aB,of as aC,Bo as aD,pf as aE,mf as aF,li as aG,vs as aH,Ms as aI,af as aJ,Fs as aK,lf as aL,df as aM,cf as aN,uf as aO,hf as aP,ff as aQ,gf as aR,wf as aS,yf as aT,bf as aU,Sf as aV,Ef as aW,If as aX,kf as aY,pn as aZ,Tf as a_,Sc as aa,jh as ab,ii as ac,Hh as ad,Yo as ae,Ss as af,Kh as ag,Es as ah,Jh as ai,ks as aj,Xh as ak,ai as al,Zh as am,Ts as an,dn as ao,Td as ap,Yh as aq,vd as ar,Md as as,tf as at,Qh as au,As as av,xs as aw,ef as ax,gn as ay,nf as az,sh as b,rd as b$,$s as b0,xf as b1,vf as b2,Gl as b3,Mf as b4,mn as b5,ci as b6,Ff as b7,$f as b8,Rf as b9,zf as bA,jf as bB,Ad as bC,Hf as bD,Ns as bE,Ic as bF,bc as bG,Sr as bH,Kf as bI,Xf as bJ,Zf as bK,Yf as bL,br as bM,St as bN,wn as bO,ed as bP,Qf as bQ,td as bR,wi as bS,od as bT,nd as bU,sd as bV,id as bW,ad as bX,ld as bY,cd as bZ,ud as b_,Rs as ba,Yi as bb,Df as bc,Nf as bd,Ct as be,_f as bf,Bf as bg,Of as bh,ui as bi,hi as bj,fi as bk,zl as bl,Pf as bm,Lf as bn,Cf as bo,Uf as bp,Un as bq,di as br,oi as bs,Wf as bt,gi as bu,mi as bv,pi as bw,qf as bx,Vf as by,Gf as bz,Ll as c,Pc as c$,yn as c0,hd as c1,Ds as c2,_s as c3,Ei as c4,fd as c5,dd as c6,gd as c7,pd as c8,bn as c9,we as cA,ol as cB,te as cC,Wo as cD,Ut as cE,uc as cF,xr as cG,fe as cH,zi as cI,Rc as cJ,sa as cK,xg as cL,rt as cM,Jc as cN,Xc as cO,Dg as cP,tl as cQ,ug as cR,Eg as cS,Tg as cT,st as cU,ke as cV,yr as cW,Me as cX,he as cY,Rg as cZ,_g as c_,yi as ca,md as cb,wd as cc,Y as cd,xe as ce,Jf as cf,bi as cg,yd as ch,bd as ci,Ii as cj,Id as ck,Sd as cl,Ed as cm,kd as cn,ve as co,Si as cp,xd as cq,Ca as cr,vg as cs,P as ct,Kc as cu,Ng as cv,Ml as cw,$e as cx,zo as cy,T as cz,Cl as d,Jt as d$,Ja as d0,ie as d1,cr as d2,qc as d3,Zs as d4,Le as d5,Mg as d6,Mc as d7,Fc as d8,kn as d9,tg as dA,Ud as dB,Od as dC,Cd as dD,it as dE,Wd as dF,dt as dG,Ha as dH,rl as dI,Ld as dJ,Pd as dK,Nt as dL,Qd as dM,Gd as dN,M as dO,tu as dP,Bg as dQ,eu as dR,Og as dS,gr as dT,Pg as dU,Dc as dV,Nc as dW,Ni as dX,el as dY,Jd as dZ,Xd as d_,ur as da,Ua as db,Or as dc,fc as dd,dr as de,hc as df,oc as dg,zc as dh,Wa as di,Ga as dj,Z as dk,Qc as dl,eg as dm,Xo as dn,hr as dp,D as dq,Zi as dr,lg as ds,Nd as dt,Ws as du,tt as dv,pg as dw,Bc as dx,Ae as dy,ra as dz,Ul as e,Gg as e$,Lg as e0,Ba as e1,Cg as e2,Qt as e3,ou as e4,iu as e5,Ag as e6,j as e7,pr as e8,na as e9,jn as eA,hg as eB,fg as eC,Ug as eD,Wg as eE,cs as eF,Tr as eG,un as eH,Fg as eI,ge as eJ,de as eK,Ko as eL,Js as eM,Ks as eN,Yn as eO,Cc as eP,Ee as eQ,Xi as eR,fs as eS,De as eT,Go as eU,Br as eV,Ne as eW,qi as eX,at as eY,ea as eZ,dl as e_,Zt as ea,$c as eb,Lc as ec,Oc as ed,vl as ee,xl as ef,zs as eg,On as eh,dc as ei,mr as ej,Bn as ek,je as el,og as em,er as en,ua as eo,Oe as ep,pc as eq,xn as er,gc as es,mc as et,rg as eu,wc as ev,yc as ew,Sn as ex,Tt as ey,il as ez,Wl as f,Sg as f$,zg as f0,hu as f1,ig as f2,kg as f3,Ht as f4,qg as f5,gs as f6,Ys as f7,vt as f8,Zo as f9,Zg as fA,qd as fB,vu as fC,Mu as fD,Fu as fE,$u as fF,Hd as fG,eo as fH,$g as fI,Zd as fJ,Ir as fK,Fd as fL,Ti as fM,sg as fN,ng as fO,Sl as fP,kl as fQ,Kd as fR,ag as fS,Ho as fT,An as fU,oe as fV,Yg as fW,Pt as fX,Nu as fY,mg as fZ,Gc as f_,pt as fa,du as fb,wl as fc,Vg as fd,zd as fe,an as ff,jg as fg,cg as fh,_ as fi,wu as fj,Hg as fk,Po as fl,Kg as fm,yg as fn,bg as fo,Ig as fp,Jg as fq,le as fr,W as fs,Re as ft,Pr as fu,Hn as fv,bl as fw,Ar as fx,Wc as fy,Ra as fz,_e as g,gt as g$,jd as g0,Yd as g1,op as g2,Fl as g3,is as g4,Ya as g5,_u as g6,Qg as g7,F as g8,Vo as g9,ze as gA,Bd as gB,z as gC,Zu as gD,Vd as gE,zt as gF,me as gG,Yu as gH,ko as gI,Xg as gJ,Eo as gK,sn as gL,oo as gM,Qu as gN,Rd as gO,$d as gP,Be as gQ,Dd as gR,wr as gS,dg as gT,gg as gU,gl as gV,It as gW,Ql as gX,wg as gY,Kn as gZ,cn as g_,Bu as ga,tp as gb,ac as gc,cc as gd,Ou as ge,Pu as gf,Lu as gg,vc as gh,Yt as gi,ep as gj,Dt as gk,np as gl,Jo as gm,sp as gn,Al as go,zu as gp,Ka as gq,Vu as gr,Hu as gs,Ku as gt,jt as gu,Xu as gv,rp as gw,Lo as gx,hl as gy,_d as gz,rh as h,oh as i,ih as j,ah as k,lh as l,ch as m,uh as n,hh as o,dh as p,fh as q,gh as r,mh as s,wh as t,ph as u,bh as v,Ih as w,ys as x,Eh as y,Sh as z};
