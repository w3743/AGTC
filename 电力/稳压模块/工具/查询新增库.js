const a=await eda.lib_Device.getByLcscIds(['C2866750']);
const b=await eda.lib_Device.search('RC0603FR-07680RL',undefined,undefined,undefined,5,1);
return {tmux:a,r680:b};
