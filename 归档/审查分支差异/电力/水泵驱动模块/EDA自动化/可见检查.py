from eda import *
VIS='77a03eaa-9e5f-48f1-a682-903cd78a315e'
code="await eda.dmt_EditorControl.openDocument('bafe8a48e4625018');return await eda.sch_Drc.check(true,false,true);"
print(cli('invoke','--session',VIS,'--ext-uuid','eda','--code',code))
