return (await eda.sch_PrimitiveWire.getAll()).filter(w=>w.getState_Net().toUpperCase().includes('FAULT')).map(w=>({net:w.getState_Net(),line:w.getState_Line()}));
