import fs from 'node:fs';
import path from 'node:path';
const root=path.resolve('artifacts/violas_final');
const out=path.join(root,'three_experiments','report');fs.mkdirSync(out,{recursive:true});
const datasets=['caltech101','cub200','flowers102','20news','ohsumed','yahoo'];
const names={caltech101:'Caltech101',cub200:'CUB200',flowers102:'Flowers102','20news':'20News',ohsumed:'OHSUMED',yahoo:'Yahoo'};
const labels={vg_128:'VG+HDMG native',vg_embedding_stable:'VG-Emb + ties',vg_embedding_index:'VG-Emb (HNSW)',vg_32:'HDMG 32簇',vg_64:'HDMG 64簇',vg_256:'HDMG 256簇',vg_exact:'VG精确扫描',vg_hdmg:'VG+HDMG',vg_rep:'VG−HDMG',milvus:'Milvus',qdrant:'Qdrant',chroma:'Chroma'};
const dbs=['milvus','qdrant','chroma'], systems=['vg_hdmg',...dbs], qualitySystems=['vg_hdmg','vg_rep',...dbs];
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const frozen=Object.fromEntries(datasets.map(d=>[d,read(`${d}/results.json`)]));
const indexed=Object.fromEntries(datasets.map(d=>[d,read(`postfilter_ann/${d}/results.json`)]));
const integrated=Object.fromEntries(datasets.map(d=>[d,read(`integrated_v6/${d}/results.json`)]));
const yahooSearchPath='yahoo_search_optimized/results.json';
const yahooSearch=fs.existsSync(path.join(root,yahooSearchPath))?read(yahooSearchPath):null;
if(yahooSearch&&yahooSearch.config.workload_sha256!==frozen.yahoo.config.workload_sha256)throw Error('Yahoo workload mismatch');
const maintenance=Object.fromEntries(datasets.map(d=>[d,Object.fromEntries(systems.map(m=>[m,read(`${m==='vg_hdmg'?'maintenance_v6':'maintenance_v2'}/${d}/${m}.json`)]))]));
for(const d of datasets){if(integrated[d].rows.length!==100||integrated[d].config.workload_sha256!==frozen[d].config.workload_sha256)throw Error('workload mismatch');}
const esc=v=>String(v).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const fmt=(v,n=4)=>v==null?'—':Number(v).toFixed(n);
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const median=a=>{a=[...a].sort((x,y)=>x-y);return (a[Math.floor((a.length-1)/2)]+a[Math.floor(a.length/2)])/2;};
const table=(heads,rows)=>['| '+heads.join(' | ')+' |','| '+heads.map(()=> '---').join(' | ')+' |',...rows.map(r=>'| '+r.join(' | ')+' |')].join('\n');
const htmlTable=(heads,rows)=>'<div class="scroll"><table><thead><tr>'+heads.map(v=>'<th>'+esc(v)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+esc(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>';
const sections=['# Violas：四组实验结果'];const html=[];
function text(s){sections.push(s);html.push('<p>'+esc(s)+'</p>');}
function heading(s){sections.push('## '+s);html.push('<h2>'+esc(s)+'</h2>');}
function addTable(heads,rows){sections.push(table(heads,rows));html.push(htmlTable(heads,rows));}
const figures=[];
function chart(file,title,panels,columns,{quality=false,speedup=false,precision=2,qualityLabel="平均 objective ratio"}={}){
 const vals=panels.flatMap(p=>p.data.flat()).filter(v=>v!=null);
 if(vals.some(v=>!Number.isFinite(v)))throw Error('invalid number');
 const logs=vals.map(v=>Math.log10(Math.max(1e-9,v)));
 const lo=Math.min(...logs),hi=Math.max(...logs),bound=Math.max(Math.abs(lo),Math.abs(hi),.01);
 const w=1160,h=90+panels.length*335,left=160,cw=950/columns.length;
 let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"><title>${esc(title)}</title><rect width="100%" height="100%" fill="white"/><style>text{font-family:Arial,'Microsoft YaHei',sans-serif;fill:#19334d}.val{font-size:17px;font-variant-numeric:tabular-nums}.small{font-size:14px}</style><text x="25" y="32" font-size="23" font-weight="600">${esc(title)}</text>`;
 const mix=(a,b,t)=>'#'+a.map((v,i)=>Math.round(v+(b[i]-v)*Math.max(0,Math.min(1,t))).toString(16).padStart(2,'0')).join('');
 panels.forEach((p,j)=>{const top=55+j*335;svg+=`<text x="25" y="${top+18}" font-size="18" font-weight="600">${esc(p.title)}</text>`;
 columns.forEach((c,i)=>svg+=`<text class="small" x="${left+(i+.5)*cw}" y="${top+44}" text-anchor="middle">${esc(labels[c]||c)}</text>`);
 datasets.forEach((d,r)=>{const y=top+58+r*40;svg+=`<text class="small" x="145" y="${y+25}" text-anchor="end">${names[d]}</text>`;
 p.data[r].forEach((v,c)=>{let norm=v==null?0:quality?v:speedup?Math.log10(Math.max(v,1e-9))/bound:(Math.log10(Math.max(v,1e-9))-lo)/(hi-lo||1);
 const color=v==null?'#e8edf2':speedup?mix([248,250,246],norm>=0?[31,132,86]:[196,56,52],Math.abs(norm)):quality?mix([237,247,252],[20,84,134],norm):mix([255,247,236],[164,36,30],norm);
 const value=v==null?'N/A':quality?fmt(v,4):speedup?fmt(v,2)+'×':fmt(v,precision);
 svg+=`<rect x="${left+c*cw}" y="${y}" width="${cw}" height="40" fill="${color}" stroke="white" stroke-width="2"/><text class="val" x="${left+(c+.5)*cw}" y="${y+26}" text-anchor="middle" style="fill:${v!=null&&Math.abs(norm)>.7?'white':'#19334d'}">${value}</text>`;
 });});});
 svg+=`<text class="small" x="25" y="${h-15}">${esc(quality?'颜色为0–1；数值为'+qualityLabel+'。':speedup?'VDB latency / 图题所示VG方法latency；大于1表示VG更快。颜色使用log10。':'数值为实际时间；颜色共用对数刻度；N/A不参与色阶。')}</text></svg>`;
 fs.writeFileSync(path.join(out,file),svg);figures.push(file);sections.push(`![${title}](${file})`);html.push(`<img src="${file}" alt="${esc(title)}">`);
}
text('四个实验分别为：带语义/硬条件与多样性的检索、建库与CRUD及独立建索引、纯embedding检索，以及旧实验一的软entity与key路由对比。实验四沿用旧软entity查询并使用当前VG实现重测五种方法，其设置与硬entity实验分别说明。六个数据集每组检索各100条冻结查询。带条件检索质量报告 objective ratio；纯embedding质量报告 Recall@3。查询时延报告 mean 和 p50。');
heading('1. 数据与环境');
text('范围：Caltech101排除BACKGROUND_Google、Faces、Faces_easy后为99类；CUB200和Flowers102沿用已完成标注的100类子集，文本数据集分别为20、23、10类。下表是实际参与检索的gallery规模，不代表原始数据集总规模。');
addTable(['数据集','Gallery对象','Entity','维数','新VG微簇','查询数'],datasets.map(d=>{const a=integrated[d].data;return[names[d],a.objects,a.entities,a.dimensions,a.clusters,100];}));
text('云端：Intel Xeon Platinum 8350C @ 2.60GHz，16逻辑CPU，62.79GiB内存，Linux 6.8.0-107-generic，Python 3.12.3。单查询并发，runner的OPENBLAS/OMP/MKL线程均为1，不进行主动预热。embedding、原全量prototype facet、intent与Oracle准备不计入检索时延。');
text('Milvus Lite 3.2.2 / PyMilvus 3.0.1：cosine IVF_FLAT，nlist=128，nprobe=16；entity INVERTED，硬facet使用ARRAY_CONTAINS_ALL。Qdrant Server/client 1.20.0：cosine HNSW，m=16，ef_construct=100，ef=80，entity和facet payload索引。Chroma 1.5.9 PersistentClient：默认cosine HNSW与metadata过滤。VG进程内运行，Milvus/Chroma为本地嵌入式，Qdrant为本机回环服务；这些部署开销保留在结果中。');
text('新VG库使用同一批单位embedding与entity/facet记录，几何聚类alpha=0.5，random_state=42，KMeans n_init=1；使用批量余弦建图。调整后的索引重新验证了同一组100条查询，未修改查询、奖励或Oracle。VG与纯embedding四系统均重新测一轮；第一组三个VDB沿用已完成结果：前过滤与带过滤索引的Caltech/CUB为3轮，其余1轮，索引后过滤全部1轮。');
heading('2. 实验一：带条件与多样性的检索');
text('每个数据集100条固定image/text embedding + intent查询，来自test对象；Gallery来自train。工作负载包含E、H1、H2、D、H+D各20条：仅entity、1条硬facet、2条硬facet、多样性、硬facet与多样性组合。本轮entity均为硬约束，未据此验证soft entity跨类检索。语义权重0.15/0.35/0.55各34/33/33条，embedding权重为其补数；lambda取0/0.05/0.10/0.15。');
text('单对象基础分 s(x) = w_sem·semantic(entity_q, entity_x) + (1−w_sem)·(1+cos(q,x))/2。集合目标 F(S) = Σ s(x) + Σ_v lambda_v·min(count_v(S), quota_v)，且 |S|≤3，所有对象必须满足硬条件。没有多样性要求时奖励项为0；同一硬entity内语义项相同。Oracle用全体合格对象精确最大化这一目标；平均ratio = mean_q[F(S_method,q)/F(S_oracle,q)]。');
text('三种VDB管线：前过滤枚举全部entity与硬facet匹配对象，再完整SELECT；后过滤无条件IVF/HNSW召回10个，再硬过滤与完整SELECT；带过滤索引在库内带entity和硬facet检索10个，再完整SELECT。最终最多返回3个，不补齐不合格对象。所有方法使用相同基础分与facet quota奖励；Oracle枚举同一硬条件范围的全部对象并精确SELECT。');
text('VG+HDMG使用真实硬条件交集定位可用微簇、4个gallery成员代表辅助排序、HDMG导航与每4簇批量评分，最多展开32簇。这里报告预算受限模式，不使用全范围认证补扫。VG−HDMG基础扫描也使用32簇上限，逐向量在线余弦；其与VG+HDMG的差异包括实现与目录使用方式，不能把全部速度差仅归因于图边。');
const modes={pre:'前过滤',post:'后过滤（索引召回）',index:'带过滤索引'};
const qmetric=(d,mode,m)=>d==='yahoo'&&mode==='post'&&yahooSearch?.aggregate[m]?yahooSearch.aggregate[m]:m.startsWith('vg_')?integrated[d].aggregate[m]:(mode==='post'?indexed[d]:frozen[d]).aggregate[`${mode}_${m}`];
for(const [tag,title,field] of [['objective','Objective ratio','objective_ratio'],['mean','Mean latency (ms)','mean_ms'],['p50','p50 latency (ms)','p50_ms']]){
 chart(`quality-${tag}.svg`,title,Object.entries(modes).map(([mode,t])=>({title:t,data:datasets.map(d=>qualitySystems.map(m=>tag==='objective'?qmetric(d,mode,m)[field]:qmetric(d,mode,m).latency[field]))})),qualitySystems,{quality:tag==='objective'});
}
for(const field of ['mean_ms','p50_ms'])chart(`quality-speedup-${field}.svg`,`${field.split('_')[0]} 加速比`,Object.entries(modes).map(([mode,t])=>({title:t,data:datasets.map(d=>dbs.map(m=>qmetric(d,mode,m).latency[field]/qmetric(d,mode,'vg_hdmg').latency[field]))})),dbs,{speedup:true});
for(const [mode,title]of Object.entries(modes)){heading(`实验一完整结果：${title}`);addTable(['数据集','方法','Objective ratio','Mean ms','p50 ms'],datasets.flatMap(d=>qualitySystems.map(m=>{const v=qmetric(d,mode,m);return[names[d],labels[m],fmt(v.objective_ratio,6),fmt(v.latency.mean_ms),fmt(v.latency.p50_ms)];})));}
if(yahooSearch){
 text('Yahoo后过滤表中的VG、Qdrant和Chroma采用最新100条冻结查询×5轮交错计时，每方法500次，无主动预热，首轮调用开销计入。VG预算32簇、原评分与硬条件均沿用既有配置，Qdrant/Chroma继续无条件召回10个后过滤。VG仍为原生带条件搜索，表中后过滤指VDB管线。VG通过批量更新导航奖励、合并无需多样性反馈的评分批次，以及预算覆盖全部合格簇时一次评分来降低开销。58条查询使用全部合格簇批量路径，其余查询保留导航。500次逐查询核验确认优化前后返回对象、objective、评分对象数与访问簇集合一致。');
 const a=yahooSearch.aggregate;
 text(`同场原版VG mean ${fmt(a.vg_before.latency.mean_ms)} ms、p50 ${fmt(a.vg_before.latency.p50_ms)} ms；优化后mean ${fmt(a.vg_hdmg.latency.mean_ms)} ms、p50 ${fmt(a.vg_hdmg.latency.p50_ms)} ms。五轮各轮的VG mean均低于Qdrant和Chroma。其余数据集、Milvus和VG−HDMG仍使用原轮次。`);
 sections.push('Yahoo优化原始记录：[500次逐查询结果](../../yahoo_search_optimized/results.json)。');
}
heading('3. 实验二：建库、CRUD与独立建索引');
text('建库从空库开始，计入数据写入、结构/索引构建与就绪。VG包括几何聚类、HDMG、语义目录、代表点和持久化快照。CRUD使用独立90% gallery副本，按entity留出约10%；单条插入、更新、删除各10个对象，按ID读取10次。更新同时修改向量与facet metadata；原零向量的更新确定性替换为单位基向量，所有生成更新向量检查有限值。批量插入、更新、删除使用全部留出对象。');
text('R是按ID读取向量与metadata，向量检索在实验一和三单独展示。单条列统计API写入完成耗时，操作后核验可见性；批量列包含索引就绪检查。Qdrant就绪轮询间隔20ms。VG操作记录写入WAL并fsync，恢复后核对ID、向量和metadata；内存向量容量预留25%，计入建库开销；倒排表只更新变化部分。另做六个数据集共480次维护后查询与全量精确SELECT的目标值一致性检查，并核对倒排表、微簇成员和几何边界。当前未做持续运行到预留容量耗尽的扩容压力测试。删除为即时逻辑删除，保留有效的保守几何上界与路由边，物理整理不计入；该策略在报告中明确区别于整图重建。');
text('独立建索引从已有向量/组开始，计入向量索引及相关metadata索引构建与持久化/就绪；VG不重复计入聚类。Chroma的本次原生接口将HNSW建设与写入结合，独立建索引标为N/A，其整体开销计在建库与插入中。此项不可据N/A声称优于Chroma。');
text('Chroma拆分核验（安装版本1.5.9、PersistentClient/Rust后端）：用独立库写入1000条64维向量，尝试把batch_size和sync_threshold提高到100000，再降低阈值。参数修改成功，但未形成可验证的“纯写入后统一建图”阶段；本地get_indexing_status接口明确返回NotImplementedError。对应版本Rust源码先向HNSW加入节点，再根据sync_threshold决定落盘，因此阈值修改耗时不能填写为建索引耗时。本次保留N/A，表示当前版本的原生接口无法按此口径单独测量；N/A不是0，也不是没有索引。Chroma的索引开销已包含在完整建库与插入耗时中，不能用该N/A计算独立建索引加速比。');
sections.push('拆分核验：[云端实测记录](../../chroma_index_separation/probe.json)；[Chroma 1.5.9 HNSW实现](https://github.com/chroma-core/chroma/blob/1.5.9/rust/segment/src/local_hnsw.rs#L736-L795)。');
html.push('<p>拆分核验：<a href="../../chroma_index_separation/probe.json">云端实测记录</a>；<a href="https://github.com/chroma-core/chroma/blob/1.5.9/rust/segment/src/local_hnsw.rs#L736-L795">Chroma 1.5.9 HNSW实现</a>。</p>');
const maintValue=(d,m,metric)=>{const r=maintenance[d][m];return metric==='build'?r.build.ready_ms/1000:metric==='index'?(r.index_build ? r.index_build.ready_ms/1000 : null):metric.startsWith('single_')||metric==='point_read'?mean(r[metric].map(v=>v.ack_ms)):r[metric].ready_ms/1000;};
chart('maintenance-build.svg','建库与独立建索引（秒）',[{title:'空库 → 可用库',data:datasets.map(d=>systems.map(m=>maintValue(d,m,'build')))},{title:'已有向量/组 → 新索引就绪',data:datasets.map(d=>systems.map(m=>maintValue(d,m,'index')))}],systems);
chart('maintenance-single.svg','单条操作 mean（ms）',['point_read','single_insert','single_update','single_delete'].map((k,i)=>({title:['R 按ID读取','C 插入','U 更新向量与metadata','D 逻辑删除'][i],data:datasets.map(d=>systems.map(m=>maintValue(d,m,k)))})),systems,{precision:4});
chart('maintenance-batch.svg','批量操作就绪耗时（秒）',['batch_insert','batch_update','batch_delete'].map((k,i)=>({title:['批量插入','批量更新','批量删除'][i],data:datasets.map(d=>systems.map(m=>maintValue(d,m,k)))})),systems);
addTable(['数据集','方法','建库秒','新建索引秒','R mean ms','C mean ms','U mean ms','D mean ms','批量对象数','批插秒','批更秒','批删秒'],datasets.flatMap(d=>systems.map(m=>{const r=maintenance[d][m];return[names[d],labels[m],...['build','index','point_read','single_insert','single_update','single_delete'].map(k=>fmt(maintValue(d,m,k),4)),r.batch_size,...['batch_insert','batch_update','batch_delete'].map(k=>fmt(maintValue(d,m,k),4))];})));
addTable(['数据集','方法','R p50 ms','C p50 ms','U p50 ms','D p50 ms'],datasets.flatMap(d=>systems.map(m=>[names[d],labels[m],...['point_read','single_insert','single_update','single_delete'].map(k=>fmt(median(maintenance[d][m][k].map(v=>v.ack_ms)),4))])));
heading('4. 实验三：纯embedding检索');
const embedding=Object.fromEntries(datasets.map(d=>[d,read(`pure_embedding_final/${d}/results.json`)]));
for(const d of datasets){if(embedding[d].rows.length!==100||embedding[d].config.workload_sha256!==frozen[d].config.workload_sha256)throw Error('pure workload mismatch');}
const pureSystems=['vg_embedding_stable',...dbs];
const diagnostics=Object.fromEntries(datasets.map(d=>[d,read(`pure_embedding_native_m32/${d}/results.json`)]));
const pure=(d,m)=>embedding[d].aggregate[m];
text('纯embedding质量改为严格Recall@3 = |返回ID集合 ∩ 全库精确top3 ID集合| / 3，每数据集100条取平均。GT用float64全库cosine计算，完全同分时按object ID排序；相似度接近但ID不同仍视为未召回。此处不再报告objective ratio。');
text('完全不使用entity条件、facet、语义权重、多样性奖励或SELECT。测试输入沿用冻结test embedding，Gallery和top3不变。本轮全部方法只返回ID/分数；三个VDB关闭向量回传，参数仍为Milvus IVF nlist128/nprobe16，Qdrant HNSW ef80，Chroma原有默认HNSW。每方法一轮100条，按查询轮换执行顺序，无主动预热，Faiss和原生VG均单线程。本节展示专用路径和三个VDB；数值沿用此前五种方法轮换执行的正式实测，本次仅调整展示，不重新计时。');
text('VG-Emb + ties是新增的纯embedding专用路径：在原gallery旁建立对象级Faiss 1.15.1 IndexHNSWFlat，所有数据集统一M=32、efConstruction=200、efSearch=256、seed=42、单线程。先检索4个对象并用float64 cosine重排；若第3、4名同分（容差1e-12），转入原生全库精确扫描消歧，再按ID稳定排序返回top3。消歧耗时完整计入。该路径绕过HDMG；带条件检索仍使用实验一的VG+HDMG。其性能收益不能归因于HDMG，也不代表提出了新的HNSW算法。');
chart('embedding.svg','纯embedding Recall@3',[{title:'严格ID Recall@3',data:datasets.map(d=>pureSystems.map(m=>pure(d,m).recall_at_3))}],pureSystems,{quality:true,qualityLabel:'平均 Recall@3'});
chart('embedding-latency.svg','纯embedding时延（ms）',['mean_ms','p50_ms'].map(k=>({title:k,data:datasets.map(d=>pureSystems.map(m=>pure(d,m).latency[k]))})),pureSystems);
addTable(['数据集','方法','Recall@3','Mean ms','p50 ms'],datasets.flatMap(d=>pureSystems.map(m=>{const r=pure(d,m);return[names[d],labels[m],fmt(r.recall_at_3,6),fmt(r.latency.mean_ms),fmt(r.latency.p50_ms)];})));
chart('embedding-speedup.svg','VG-Emb 专用路径对VDB的加速比',['mean_ms','p50_ms'].map(k=>({title:k,data:datasets.map(d=>dbs.map(m=>pure(d,m).latency[k]/pure(d,'vg_embedding_stable').latency[k]))})),dbs,{speedup:true});
text('上述专用路径为进程内Faiss索引，Qdrant包含本机服务调用，Milvus/Chroma包含各自客户端和存储层开销。加速比是所测部署的端到端检索差异，不等于纯索引算法速度差；mean包含首次调用和自然缓存影响，需同时看p50。');
heading('纯embedding同分诊断');
addTable(['数据集','100条中零向量查询','触发同分精确扫描','原始HNSW Recall@3（诊断轮）','消歧路径 Recall@3（正式轮）'],datasets.map(d=>[names[d],embedding[d].rows.filter(r=>r.query_norm===0).length,pure(d,'vg_embedding_stable').tie_fallbacks,fmt(diagnostics[d].aggregate.vg_embedding_index.recall_at_3,6),fmt(pure(d,'vg_embedding_stable').recall_at_3,6)]));
text('并列分数会使多个不同ID集合具有相同相似度，严格ID recall可能低于距离意义上的质量。因此不能仅凭上述差距断言VDB返回的相似度更差。零向量查询若存在，将所有cosine约定为0并按ID选GT，只为口径确定，不视作有意义的语义近邻请求；所有100条均保留。消歧不能保证补回HNSW遗漏的非并列近邻，其Recall仍按实际结果测量。');
heading('纯embedding新增索引成本');
text('VG-Emb增加一份对象级索引。下面单独列出其从已有单位向量构建并持久化的耗时和文件大小，不能将实验二的原VG建库耗时直接当成启用此路径后的完整成本。两阶段相加是成本估计，不是联合建库实测；本轮没有测量启用辅助索引后的双索引CRUD。');
addTable(['数据集','原VG建库秒','额外HNSW构建秒','两阶段耗时之和秒','额外索引MiB'],datasets.map(d=>{const r=embedding[d],base=maintenance[d].vg_hdmg.build.ready_ms;return[names[d],fmt(base/1000),fmt(r.auxiliary_index_build_ms/1000),fmt((base+r.auxiliary_index_build_ms)/1000),fmt(r.auxiliary_index_bytes/1048576,2)];}));
sections.push(fs.readFileSync(path.join(out,'experiment4.md'),'utf8').trim().replace(/^# 实验四：/,'## 5. 实验四：').replace(/^## (实验设置|结果图|完整结果|原始结果)$/gm,'### $1'));
heading('6. 指标与证据边界');
text('Objective ratio用于实验一的带条件与多样性检索，衡量实际返回集合达到该查询最优目标值的比例。每条查询先计算一个比值，再按查询等权平均。实验三的纯embedding检索使用严格ID Recall@3，其定义见实验三。');
text('（1）确定查询的合格对象范围。记查询为q，Gallery为G，满足required_entity以及全部hard_filters的对象集合为E_q。本轮entity为硬约束，多个硬facet条件取交集。最终返回集合S必须满足S⊆E_q且|S|≤k，本轮k=3。后过滤先无条件召回10个候选，再去掉不合格对象并SELECT，最多返回3个；候选被过滤后可以少于3个。');
text('（2）计算单个对象的基础分。设q与对象x的embedding余弦相似度为cos(q,x)，entity查询与对象entity的单位语义向量分别为u_q与u_x。当前实现将embedding相似度映射为e(q,x)=(1+cos(q,x))/2，将entity相似度映射并裁剪为t(q,x)=clip((1+u_q·u_x)/2,0,1)。基础分按下式计算。');
sections.push(String.raw`$$
s_q(x)=w_{\mathrm{emb}}\,e(q,x)+w_{\mathrm{sem}}\,t(q,x),\qquad
w_{\mathrm{emb}}+w_{\mathrm{sem}}=1.
$$`);
text('权重读取每条查询的resolved_plan，本轮w_sem为0.15、0.35或0.55，w_emb为其补数。比如cos(q,x)=0.8、t(q,x)=1、w_sem=0.35，则基础分为0.65×0.9+0.35×1=0.935。同一硬entity内的语义项相同，完整目标值及其比值仍包含这一项。');
text('（3）计算整个返回集合的目标值。每条查询至多使用一个facet_dimension作为多样性维度。对该维度的类别值v，n_v(S)表示集合内属于v的对象数，lambda_v表示每个受奖励对象的加分，quota_v表示该类别最多有多少个对象可获得奖励。');
sections.push(String.raw`$$
F_q(S)=\sum_{x\in S}s_q(x)+\sum_v\lambda_v\min\bigl(n_v(S),\mathrm{quota}_v\bigr).
$$`);
text('类别奖励来自resolved_plan.category_preferences中的lambda和quota。例如某类别lambda=0.10、quota=1，则该类别出现1、2或3个对象时，类别总奖励都为0.10。未指定偏好的类别、未知类别以及未设置多样性维度的查询，相关奖励为0。没有多样性奖励时，F_q(S)就是返回对象基础分之和。结果不足3个时按实际返回集合计分，漏掉的对象贡献及相关奖励会反映在总目标值中。');
text('（4）构造同一查询的Oracle。对完整E_q中的对象计算相同基础分，在相同k、facet维度、lambda和quota下求最优集合。Oracle的搜索范围覆盖全部合格Gallery对象，独立于某个方法召回的10个候选或VG访问的微簇。当前实现按类别将对象按基础分降序排列，再用堆合并各类别的边际收益序列；非负奖励和截断quota使每个类别的边际收益递减，从而精确求解当前单维度目标。');
sections.push(String.raw`$$
S_q^*\in\operatorname*{arg\,max}_{S\subseteq E_q,\ |S|\le k}F_q(S),\qquad
R_{q,m}=\frac{F_q(S_{q,m})}{F_q(S_q^*)}.
$$`);
text('其中m为被测方法，S_{q,m}为它的实际合格返回集合。各方法共用冻结Oracle，计算Oracle的准备时间在检索计时之外。分母大于0时，理论上R在0到1之间，R=1表示达到该目标函数的最优值；若返回集合为空，则F_q(∅)=0、R=0。不同对象集合可能具有相同目标值，因此相同ratio可以对应不同ID。R同时受基础分、返回数量和类别奖励影响，较高ratio表示集合目标损失较小。');
text('（5）数值示例。假设Oracle选出3个对象，基础分分别为0.90、0.85、0.80，类别分别为A、A、B；A与B都设置lambda=0.10、quota=1。Oracle目标值为0.90+0.85+0.80+0.10+0.10=2.75。某方法返回的3个对象基础分为0.90、0.85、0.70，类别全为A，则目标值为0.90+0.85+0.70+0.10=2.55，该查询的ratio为2.55/2.75≈0.927273，即获得约92.73%的最优目标值。此例仅用于解释计算方式。');
text('（6）聚合成表格指标。每个数据集有N=100条冻结查询。若每条查询执行T轮，则表中统计所有查询与轮次的单次ratio算术平均，公式如下；T=1时即100个查询ratio的平均值。');
sections.push(String.raw`$$
\overline{R}_m=\frac{1}{NT}\sum_{q=1}^{N}\sum_{r=1}^{T}
\frac{F_q(S_{q,m,r})}{F_q(S_q^*)}.
$$`);
text('该聚合给每条查询相同权重。若将所有返回目标值相加后除以所有Oracle目标值之和，查询权重会随Oracle大小变化。当前主表采用上式的逐查询平均。Yahoo最新复测T=5，共500个ratio，优化前后这500次的目标值逐项一致。六个数据集共600条冻结查询的Oracle目标值均大于0，因此当前结果没有零分母。通用结果函数遇到零分母会记录null，此类记录需要单独处理后再聚合；当前直接相除的复测脚本依赖上述正分母条件。');
text('（7）精度与解释。原始JSON保留浮点计算结果，图中objective ratio显示4位小数，完整结果表显示6位小数。Yahoo实际平均值约为0.9999848442244971，图中四舍五入为1.0000，表中为0.999985。它对应约99.998484%的最优目标值，平均相对目标损失约为0.001516%。判断是否逐查询精确达到最优需要检查原始值，显示为1.0000只说明舍入后的数值。浮点误差也可能使个别ratio略高于1，当前校验允许2×10⁻⁶的数值容差。');
sections.push('实现依据见 [对象评分与单次ratio](../../../../scripts/run_soft_entity_bench.py)、[集合目标与精确SELECT](../../../../scripts/run_guideline_pipeline.py)、[冻结基准与多轮聚合](../../../../scripts/run_final_violas_bench.py)以及[Yahoo五轮复测](../../../../scripts/benchmark_yahoo_search.py)。');
text('所有阶段使用冻结查询与标签，纯embedding阶段移除条件与奖励。本工作负载用于开发验证，调优后的同一600条查询仍需通过独立查询集进一步检验泛化能力。硬件和部署各层面的实际开销计入所测端到端延迟。');
text('CRUD覆盖现有entity上的写入、更新与删除，更新额外写入maintenance_revision标记以验证metadata变化；不把新entity扩展、并发事务、后台整理和故障恢复的完整工程能力作为已验证结论。VG已做快照/WAL正常重载一致性验证；未进行断电故障注入。不同系统后台维护和持久化策略不同，API完成与就绪检查不构成事务、耐久性或长期稳态开销完全等价的证明。索引状态检查和图库拓扑正确性核验独立于检索计时。');
heading('7. 结果解读与局限');
text('带条件检索：VG+HDMG六个数据集平均objective ratio为0.999972–1.000000，均高于本轮三个VDB的带过滤索引结果。该结果来自扩大到32簇预算、代表点辅助排序与批量评分；一些查询已覆盖全部合格微簇，不能据此宣称在所有预算和规模上都保持同样质量。');
text('维护：VG完整建库在本轮六个数据集上均快于三个VDB；按ID读取也较快。但单条插入/更新、批量写入/更新和独立建索引并非全面领先，完整结果如上，不把建库优势等同于所有CRUD都更快。');
text('纯embedding实验报告新增对象级HNSW专用路径与三个VDB的对比。更快的专用路径须结合Recall@3、额外建库成本及部署差异一起解读，不能用其结果证明HDMG本身优于现有ANN算法。参数在同一600条开发查询上验证，尚需独立测试集确认泛化。');
text('本轮单次建库/批量维护以及10次单条操作用于MVP验证，未形成跨机器、多轮统计置信区间。无主动预热不等于每次查询都是冷缓存；系统首次调用和自然缓存效果均保留，尤其需结合mean与p50解读。');
heading('8. 可追溯数据');
for(const d of datasets)sections.push(`- ${names[d]}：[纯embedding Recall@3与时延](../../pure_embedding_final/${d}/results.json)。`);
for(const d of datasets){sections.push(`- ${names[d]}：[最终检索](../../integrated_v6/${d}/results.json)；[VG维护](../../maintenance_v6/${d}/vg_hdmg.json)；[Milvus](../../maintenance_v2/${d}/milvus.json) / [Qdrant](../../maintenance_v2/${d}/qdrant.json) / [Chroma](../../maintenance_v2/${d}/chroma.json) 维护。`);}
fs.writeFileSync(path.join(out,'violas_experiment.md'),sections.join('\n\n')+'\n');
console.log(JSON.stringify({output:out,figures:figures.length,quality:datasets.map(d=>({dataset:d,vg:integrated[d].aggregate.vg_hdmg.objective_ratio,best_index:Math.max(...dbs.map(m=>frozen[d].aggregate[`index_${m}`].objective_ratio))}))}));
