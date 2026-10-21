/* Shared navigation only; all camera/depth operations remain page-owned. */
(()=>{'use strict';
 function mount(){
  const root=document.getElementById('app-shell');if(!root||root.childElementCount)return;
  const items=[['depth','深度图','/?view=depth'],['preview','相机预览','/?view=raw'],['hardware','设备状态','/hardware'],['calibration','标定','/calibration']];
  const active=root.dataset.page==='preview'&&location.pathname==='/'&&new URLSearchParams(location.search).get('view')!=='raw'?'depth':root.dataset.page;
  const header=document.createElement('div');header.className='app-header';
  const brand=document.createElement('a');brand.className='app-brand';brand.href='/?view=depth';brand.append(document.createTextNode('双目视觉'));
  const platform=document.createElement('small');platform.textContent='RK3588';brand.append(platform);
  const nav=document.createElement('nav');nav.className='app-nav';nav.setAttribute('aria-label','主导航');
  for(const [key,label,href]of items){const a=document.createElement('a');a.href=href;a.textContent=label;if(active===key)a.setAttribute('aria-current','page');nav.append(a);}
  header.append(brand,nav);root.append(header);
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',mount,{once:true});else mount();
})();

