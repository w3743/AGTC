from eda import *
r=cli('invoke','--session','850df63a-b1a0-4be9-8484-2e1e0a963d89','--ext-uuid','eda','--timeout','60000','--code',"const id=await eda.dmt_EditorControl.openDocument('bafe8a48e4625018');return {id,active:await eda.dmt_EditorControl.activateDocument(id)};")
print(r)
