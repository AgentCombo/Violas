import fs from 'node:fs';
import path from 'node:path';
const source=path.resolve('artifacts/violas_experiment/violas_experiment.md');
const out=path.resolve('artifacts/violas_final/three_experiments/report');
const datasets=['Caltech101','CUB200','Flowers102','20News','OHSUMED','Yahoo'];
const methods=['VG+HDMG','VG−HDMG','Milvus key','Qdrant key','Chroma key'];
const current=process.argv.includes('--current');
const slugs=['caltech101','cub200','flowers102','20news','ohsumed','yahoo'];
const methodIds=['vg_hdmg','vg_rep','milvus_key','qdrant_key','chroma_key'];
const evidence=[];
const rows=[];let dataset;
for(const line of fs.readFileSync(source,'utf8').split(/\r?\n/)){
  if(line.startsWith('### '))dataset=line.slice(4).trim();
  if(!datasets.includes(dataset)||!line.startsWith('|'))continue;
  const cells=line.split('|').slice(1,-1).map(s=>s.trim());
  if(cells[0]==='1'&&methods.includes(cells[1]))rows.push([dataset,cells[1],cells[2],cells[7],cells[8]]);
}
if(current){
  rows.length=0;
  datasets.forEach((d,i)=>{
    const file=`artifacts/violas_final/experiment4_current/${slugs[i]}/results.json`;
    const report=JSON.parse(fs.readFileSync(file,'utf8'));
    if(report.rows.length!==100)throw Error('Incomplete experiment');
    evidence.push(file);
    methods.forEach((m,j)=>{
      const v=report.aggregate[methodIds[j]];
      rows.push([d,m,v.objective_ratio.toFixed(6),v.latency.mean_ms.toFixed(4),v.latency.p50_ms.toFixed(4)]);
    });
  });
}
if(rows.length!==30||new Set(rows.map(r=>r.slice(0,2).join('/'))).size!==30)throw Error('Expected exactly 30 unique source rows');
for(const row of rows)if(row.slice(2).some(v=>!Number.isFinite(Number(v))))throw Error('Invalid metric');
const ordered=datasets.flatMap(d=>methods.map(m=>rows.find(r=>r[0]===d&&r[1]===m)));
fs.mkdirSync(out,{recursive:true});
const escape=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
for(const [column,key,title] of [[2,'objective','Objective ratio'],[3,'mean','Mean latency (ms)'],[4,'p50','p50 latency (ms)']]){
  const values=ordered.map(r=>Number(r[column]));
  const scale=v=>column===2?v:Math.log10(v);
  const lo=column===2?0:Math.min(...values.map(scale)),hi=column===2?1:Math.max(...values.map(scale));
  const svg=['<svg xmlns="http://www.w3.org/2000/svg" width="950" height="440" viewBox="0 0 950 440">','<rect width="950" height="440" fill="white"/>','<g font-family="Arial, sans-serif" fill="#17243b">',`<text x="24" y="34" font-size="22" font-weight="bold">Experiment 4 · ${title}</text>`];
  methods.forEach((m,j)=>svg.push(`<text x="${235+j*150}" y="78" text-anchor="middle" font-size="16">${escape(m)}</text>`));
  datasets.forEach((d,i)=>{
    svg.push(`<text x="145" y="${119+i*47}" text-anchor="end" font-size="16">${d}</text>`);
    methods.forEach((m,j)=>{
      const row=ordered.find(r=>r[0]===d&&r[1]===m),v=Number(row[column]),t=(scale(v)-lo)/(hi-lo||1);
      const color=`rgb(${Math.round(240-180*t)},${Math.round(246-135*t)},${Math.round(255-65*t)})`;
      svg.push(`<rect x="${162+j*150}" y="${91+i*47}" width="146" height="43" rx="4" fill="${color}"/><text x="${235+j*150}" y="${119+i*47}" text-anchor="middle" font-size="16" fill="${t>.7?'white':'#17243b'}">${row[column]}</text>`);
    });
  });
  svg.push(`<text x="24" y="410" font-size="13">Source: ${current?'2026-11-11 rerun, current VG, 100 queries per dataset.':'2026-11-10 report, Experiment 1.'} ${column===2?'Linear color scale; higher is better.':'Log color scale; lower is better.'}</text></g></svg>`);
  fs.writeFileSync(path.join(out,`experiment4-${key}.svg`),svg.join('\n'));
}
const md=[
'# 实验四：软 entity 场景下的 key 路由检索对比',
current?'本节于2026-11-11使用当前 VG 索引与实现重新运行五种方法。沿用 [旧报告实验一](../../../violas_experiment/violas_experiment.md) 的软 entity 查询与目标，六个数据集各100条查询，五种方法均重新计时。仅展示 objective ratio、mean 和 p50 latency。':'本节整理自 [2026-11-10 旧报告的实验一](../../../violas_experiment/violas_experiment.md)，直接保留原表数值，未重新运行实验。六个数据集各100条查询，仅展示 VG+HDMG、VG−HDMG、Milvus key、Qdrant key、Chroma key 的 objective ratio、mean 和 p50 latency。',
'## 实验设置',
'本实验使用软 entity 偏好，允许跨 entity 返回。三个 VDB 先按 entity 语义相似度选3个 key，在其对象并集中按 embedding 召回最多80个候选，再按 embedding 与软 entity 加权分数重排，最终返回最多3个对象。80是三个 key 合并后的总上限；候选数量不超过 min(80, 三个 key 的对象总数)。',
'VG 使用硬 facet 和多样性要求；本实验的 VDB key 路径不使用这些要求进行过滤或集合选择。评价统一按完整 intent 的有效返回集合与全库 Oracle 计算 objective ratio。它展示不处理完整 intent 的检索路径与完整语义检索的差距，不能单独证明现有数据库无法支持 facet，也不能将质量差距全部归因于 HDMG。',
current?'VG 使用 maintenance_v6 的新建库索引与当前 AdaptiveHDMG，两种 VG 的展开预算均为32簇，VG+HDMG每4簇批量评分，certify=False，不做预算耗尽后的认证补扫；当合格微簇数不超过预算时直接批量评分全部合格对象。VG−HDMG仍使用基础逐向量 online-cosine 扫描。所有方法每条查询执行一轮，轮换执行顺序，无主动预热，runner的OPENBLAS/OMP/MKL线程均为1。时延包含路由、召回与重排，Oracle与输入准备不计时。mean、p50单位为毫秒，首次调用和自然缓存开销保留。':'VG 沿用旧实验配置和结果，未替换为实验一至三的最新优化版本。图像采用旧 online-cosine 批次；文本 VG+HDMG 采用 verified 复测，OHSUMED 使用 CSR 核验复测，其余方法沿用原批次。时延跨批次汇总；mean、p50均为100条查询上的统计量，单位为毫秒。',
...(current?['已核对新旧 gallery 的对象ID、entity、facet和向量一致，并对全部600条查询重新计算全库 Oracle，逐条验证与冻结软 entity 目标值一致。实验四允许跨 entity；实验一使用硬 entity，两个任务的质量不能直接相互替换。表格保留六位小数，显示1.000000不代表所有查询严格达到精确最优。']:[]),
'## 结果图',
'![实验四 Objective ratio](experiment4-objective.svg)',
'![实验四 Mean latency](experiment4-mean.svg)',
'![实验四 p50 latency](experiment4-p50.svg)',
'## 完整结果',
['| 数据集 | 方法 | Objective ratio | Mean ms | p50 ms |','| --- | --- | --- | --- | --- |',...ordered.map(r=>'| '+r.join(' | ')+' |')].join('\n'),
...(current?['## 原始结果',slugs.map((s,i)=>`- ${datasets[i]}：[本轮逐查询结果](../../experiment4_current/${s}/results.json)`).join('\n')]:[])
].join('\n\n')+'\n';
fs.writeFileSync(path.join(out,'experiment4.md'),md);
fs.writeFileSync(path.join(out,'experiment4-source.json'),JSON.stringify({source:current?evidence:'artifacts/violas_experiment/violas_experiment.md',source_experiment:current?4:1,rerun:current,columns:['dataset','method','objective_ratio','mean_ms','p50_ms'],rows:ordered},null,2)+'\n');
console.log(JSON.stringify({rows:ordered.length,report:path.join(out,'experiment4.md')}));
