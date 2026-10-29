// 嘉立创EDA官方API只读结构验收。由API Bridge在EDA运行时执行；不访问文件系统、fetch或浏览器DOM。
// 运行参数：__CLI__.args.contract，或 __CLI__.args.bom（均接受参数对象或JSON字符串）。
// 必做只读网表阶段：导出EasyEDA JSON网表并逐针比较net；可选择返回原始网表文本。
return await (async function runReadOnlyPinAcceptance() {
  const DEFAULT_PAGE_UUID = "9546dca0bc8bce7f";
  const cliArgs = typeof __CLI__ !== "undefined" && __CLI__ && __CLI__.args && typeof __CLI__.args === "object"
    ? __CLI__.args
    : {};

  function parseObject(value) {
    if (typeof value !== "string") return value;
    try { return JSON.parse(value); } catch (error) {
      return { __parseError: String(error && error.message || error) };
    }
  }
  const rawInput = parseObject(cliArgs.contract !== undefined ? cliArgs.contract : (cliArgs.bom !== undefined ? cliArgs.bom : {}));
  const contract = rawInput && rawInput.contract ? rawInput.contract : rawInput;
  const components = Array.isArray(contract && contract.components) ? contract.components : [];
  const targetPageUuid = String(
    (contract && (contract.target_page_uuid || contract.targetPageUuid)) ||
    (rawInput && (rawInput.target_page_uuid || rawInput.targetPageUuid)) ||
    cliArgs.target_page_uuid || cliArgs.targetPageUuid || DEFAULT_PAGE_UUID
  );
  const expectedProjectUuid = (contract && (contract.target_project_uuid || contract.expectedProjectUuid || contract.project_uuid)) ||
    (rawInput && (rawInput.target_project_uuid || rawInput.expectedProjectUuid || rawInput.project_uuid)) || null;
  const options = Object.assign({}, (rawInput && rawInput.options) || {}, (contract && contract.options) || {}, cliArgs.options || {});

  function firstDefined(object, snake, camel) {
    if (!object) return undefined;
    return object[snake] !== undefined ? object[snake] : object[camel];
  }
  function safeState(object, method) {
    try { return object && typeof object[method] === "function" ? object[method]() : undefined; }
    catch (error) { return { __readError: String(error && error.message || error) }; }
  }
  function componentRef(item) { return String(firstDefined(item, "ref", "reference") || "").trim(); }
  function expectedPin(pin) {
    return {
      number: pin && pin.number !== undefined && pin.number !== null ? String(pin.number).trim() : "",
      name: pin && pin.name !== undefined && pin.name !== null ? String(pin.name).trim() : "",
      net: pin && pin.connection === "NC" ? "" : (pin && pin.net !== undefined && pin.net !== null ? String(pin.net) : null),
      connection: pin && pin.connection !== undefined && pin.connection !== null ? String(pin.connection).trim() : null
    };
  }
  function comparePinDefinitions(expectedSource, actualPins) {
    const expectedPins = Array.isArray(expectedSource) ? expectedSource.map(expectedPin) : [];
    if (expectedPins.length === 0) {
      return { status: "NO_EXPECTED_PINS_IN_CONTRACT", expectedCount: 0, actualCount: actualPins.length,
        missingExpected: [], unexpectedActual: actualPins, nameMismatches: [], duplicateExpectedNumbers: [], duplicateActualNumbers: [] };
    }
    const expectedNumbers = expectedPins.map(pin => pin.number).filter(Boolean);
    const actualNumbers = actualPins.map(pin => pin.number).filter(Boolean);
    const duplicates = values => [...new Set(values.filter((value, index) => values.indexOf(value) !== index))];
    const expectedByNumber = new Map(expectedPins.filter(pin => pin.number).map(pin => [pin.number, pin]));
    const actualByNumber = new Map(actualPins.filter(pin => pin.number).map(pin => [pin.number, pin]));
    const missingExpected = expectedPins.filter(pin => pin.number
      ? !actualByNumber.has(pin.number)
      : (pin.name && !actualPins.some(actual => actual.name === pin.name)));
    const unexpectedActual = actualPins.filter(pin => pin.number
      ? !expectedByNumber.has(pin.number)
      : !expectedPins.some(want => want.name && want.name === pin.name));
    const nameMismatches = expectedPins.filter(want => want.number && want.name && actualByNumber.has(want.number))
      .map(want => ({ expected: want, actual: actualByNumber.get(want.number) }))
      .filter(pair => pair.actual.name !== pair.expected.name);
    const duplicateExpectedNumbers = duplicates(expectedNumbers);
    const duplicateActualNumbers = duplicates(actualNumbers);
    const passed = missingExpected.length === 0 && unexpectedActual.length === 0 &&
      duplicateExpectedNumbers.length === 0 && duplicateActualNumbers.length === 0;
    return { status: passed ? "PASS_PIN_NUMBER_SET_MATCH" : "FAIL_PIN_NUMBER_SET_MISMATCH",
      expectedCount: expectedPins.length, actualCount: actualPins.length, missingExpected, unexpectedActual,
      nameMismatches, duplicateExpectedNumbers, duplicateActualNumbers,
      note: "引脚名称仅供人工检查；符号名可能与BOM名不同。" };
  }

  const project = await eda.dmt_Project.getCurrentProjectInfo();
  const currentDocument = await eda.dmt_SelectControl.getCurrentDocumentInfo();
  const currentPage = await eda.dmt_Schematic.getCurrentSchematicPageInfo();
  const pages = await eda.dmt_Schematic.getAllSchematicPagesInfo();
  const pageListed = Array.isArray(pages) && pages.some(page => page && page.uuid === targetPageUuid);
  const documentIsSchematicPage = !!currentDocument && currentDocument.documentType === 1;
  const pageMatches = !!currentPage && currentPage.uuid === targetPageUuid && !!currentDocument && currentDocument.uuid === targetPageUuid;
  const projectMatches = !!project && (!expectedProjectUuid || project.uuid === expectedProjectUuid) &&
    (!currentDocument || !currentDocument.parentProjectUuid || currentDocument.parentProjectUuid === project.uuid);
  const contextReady = !!project && documentIsSchematicPage && pageMatches && pageListed && projectMatches;
  const context = {
    expectedProjectUuid: expectedProjectUuid || null,
    actualProjectUuid: project && project.uuid || null,
    currentDocumentType: currentDocument && currentDocument.documentType,
    currentDocumentUuid: currentDocument && currentDocument.uuid || null,
    currentDocumentProjectUuid: currentDocument && currentDocument.parentProjectUuid || null,
    expectedPageUuid: targetPageUuid,
    currentPageUuid: currentPage && currentPage.uuid || null,
    currentPageName: currentPage && currentPage.name || null,
    targetPageFoundInProject: pageListed,
    projectMatchesExpected: projectMatches,
    currentDocumentIsSchematicPage: documentIsSchematicPage,
    currentPageMatchesTarget: pageMatches
  };
  if (!contextReady) {
    return {
      schema: "EASYEDA_READ_ONLY_PIN_ACCEPTANCE_V1",
      status: "BLOCKED_WRONG_OR_UNAVAILABLE_PROJECT_PAGE",
      readOnly: true,
      input: { source: cliArgs.contract !== undefined ? "__CLI__.args.contract" : (cliArgs.bom !== undefined ? "__CLI__.args.bom" : "none"),
        contractComponentCount: components.length, contractParseError: rawInput && rawInput.__parseError || null },
      context, referenceAudit: null, perComponent: null, netlistExport: { status: "NOT_REQUESTED" },
      notes: ["未在指定工程和图页上执行器件/引脚读取；本次没有修改工程。"]
    };
  }

  const primitives = await eda.sch_PrimitiveComponent.getAll(undefined, false);
  const actualComponents = [];
  for (const primitive of primitives || []) {
    const ref = String(safeState(primitive, "getState_Designator") || "").trim();
    if (!ref) continue; // 无位号的网络端口/电源标识不是BOM器件
    const primitiveId = safeState(primitive, "getState_PrimitiveId");
    const device = safeState(primitive, "getState_Component");
    const pinsRead = primitiveId && !primitiveId.__readError
      ? await eda.sch_PrimitiveComponent.getAllPinsByPrimitiveId(primitiveId)
      : undefined;
    const pins = (pinsRead || []).map(pin => ({
      number: String(safeState(pin, "getState_PinNumber") || "").trim(),
      name: String(safeState(pin, "getState_PinName") || "").trim(),
      noConnected: safeState(pin, "getState_NoConnected")
    }));
    actualComponents.push({ ref: ref || null, primitiveId: primitiveId && !primitiveId.__readError ? primitiveId : null,
      componentLibraryUuid: device && device.libraryUuid || null, componentUuid: device && device.uuid || null,
      pins, pinsRead: Array.isArray(pinsRead) });
  }

  const actualByRef = new Map();
  for (const actual of actualComponents) {
    if (!actual.ref) continue;
    if (!actualByRef.has(actual.ref)) actualByRef.set(actual.ref, []);
    actualByRef.get(actual.ref).push(actual);
  }
  const expectedRefs = components.map(componentRef).filter(Boolean);
  const expectedRefSet = new Set(expectedRefs);
  const actualRefSet = new Set(actualByRef.keys());
  const duplicateExpectedRefs = [...new Set(expectedRefs.filter((ref, index) => expectedRefs.indexOf(ref) !== index))];
  const duplicateActualRefs = [...actualByRef.entries()].filter(([, rows]) => rows.length > 1).map(([ref]) => ref);
  const unidentifiedActualComponentCount = actualComponents.filter(item => !item.ref).length;
  const invalidExpectedRefCount = components.filter(item => !componentRef(item)).length;
  const missingExpectedRefs = [...expectedRefSet].filter(ref => !actualRefSet.has(ref));
  const extraActualRefs = [...actualRefSet].filter(ref => !expectedRefSet.has(ref));

  const perComponent = components.map(target => {
    const ref = componentRef(target);
    const rows = actualByRef.get(ref) || [];
    const actual = rows.length === 1 ? rows[0] : null;
    const expectedPins = Array.isArray(target.pins) ? target.pins : [];
    const pinAudit = actual ? comparePinDefinitions(expectedPins, actual.pins) : {
      status: rows.length > 1 ? "AMBIGUOUS_DUPLICATE_ACTUAL_REFERENCE" : "TARGET_REFERENCE_NOT_PRESENT",
      expectedCount: expectedPins.length, actualCount: null, missingExpected: expectedPins,
      unexpectedActual: [], nameMismatches: [], duplicateExpectedNumbers: [], duplicateActualNumbers: []
    };
    return {
      ref,
      manufacturerPart: firstDefined(target, "manufacturer_part", "manufacturerPart") || null,
      supplierPart: firstDefined(target, "supplier_part", "supplierPart") || null,
      scope: target.scope || null,
      footprint: target.footprint || null,
      actualCount: rows.length,
      actualDevice: actual ? { libraryUuid: actual.componentLibraryUuid, uuid: actual.componentUuid } : null,
      expectedPins,
      actualPins: actual ? actual.pins : [],
      pinAudit
    };
  });

  let netlistExport = { status: "NOT_READ" };
  let netlistByRef = new Map();
  let netlistRefAudit = null;
  let netlistText;
  let unidentifiedNetlistComponentCount = null;
  try {
    // SCH_ManufactureData.getNetlistFile(fileName?, netlistType?)；"EasyEDA"是ESYS_NetlistType.EASYEDA_PRO的枚举值。
    const file = await eda.sch_ManufactureData.getNetlistFile(undefined, "EasyEDA");
    if (!file) {
      netlistExport = { status: "EXPORT_RETURNED_UNDEFINED", netlistType: "EasyEDA" };
    } else if (typeof file.text !== "function") {
      netlistExport = { status: "FILE_TEXT_METHOD_UNAVAILABLE", netlistType: "EasyEDA",
        fileName: file.name || null, sizeBytes: typeof file.size === "number" ? file.size : null };
    } else {
      netlistText = String(await file.text()).replace(/^\uFEFF/, "");
      const parsed = JSON.parse(netlistText);
      const netlistComponentsObject = parsed && parsed.components;
      const netlistRows = Array.isArray(netlistComponentsObject)
        ? netlistComponentsObject
        : Object.values(netlistComponentsObject || {});
      unidentifiedNetlistComponentCount = netlistRows.filter(row => !String(row && row.props && row.props.Designator || "").trim()).length;
      for (const row of netlistRows) {
        const ref = String(row && row.props && row.props.Designator || "").trim();
        if (!ref) continue;
        if (!netlistByRef.has(ref)) netlistByRef.set(ref, []);
        netlistByRef.get(ref).push(row);
      }
      const expectedRefSetForNetlist = new Set(components.map(componentRef).filter(Boolean));
      const netlistRefSet = new Set(netlistByRef.keys());
      netlistRefAudit = {
        missingExpectedRefs: [...expectedRefSetForNetlist].filter(ref => !netlistRefSet.has(ref)),
        extraNetlistRefs: [...netlistRefSet].filter(ref => !expectedRefSetForNetlist.has(ref)),
        duplicateNetlistRefs: [...netlistByRef.entries()].filter(([, rows]) => rows.length > 1).map(([ref]) => ref),
        unidentifiedNetlistComponentCount
      };
      netlistExport = {
        status: "EXPORTED_AND_PARSED",
        netlistType: "EasyEDA",
        fileName: file.name || null,
        mimeType: file.type || null,
        sizeBytes: typeof file.size === "number" ? file.size : null,
        componentRecords: netlistRows.length,
        connectivitySource: "components[uid].props.Designator + pinInfoMap[pin].net",
        rawTextIncluded: options.include_netlist_text === true || options.includeNetlistText === true,
        text: options.include_netlist_text === true || options.includeNetlistText === true ? netlistText : undefined
      };
    }
  } catch (error) {
    netlistExport = { status: "EXPORT_OR_PARSE_ERROR", error: String(error && error.message || error), netlistType: "EasyEDA" };
  }

  const netAuditByRef = new Map();
  for (const target of components) {
    const ref = componentRef(target);
    const rows = netlistByRef.get(ref) || [];
    const netlistComponent = rows.length === 1 ? rows[0] : null;
    const schematicRows = actualByRef.get(ref) || [];
    const schematicComponent = schematicRows.length === 1 ? schematicRows[0] : null;
    const schematicPinsByNumber = new Map((schematicComponent && schematicComponent.pins || []).map(pin => [pin.number, pin]));
    const pinInfoMap = netlistComponent && netlistComponent.pinInfoMap || {};
    const expectedPins = Array.isArray(target.pins) ? target.pins.map(expectedPin) : [];
    const expectedPinNumbers = expectedPins.map(pin => pin.number).filter(Boolean);
    const actualPinNumbers = Object.keys(pinInfoMap).map(String);
    const missingExpectedPins = [];
    const unexpectedActualPins = actualPinNumbers.filter(number => !expectedPinNumbers.includes(number));
    const pinNetMismatches = [];
    const pinNCFlagMismatches = [];
    const missingExpectedNets = [];
    const pinConnectionRows = [];
    for (const expected of expectedPins) {
      if (!expected.number) {
        missingExpectedPins.push({ expected, reason: "CONTRACT_PIN_NUMBER_MISSING" });
        continue;
      }
      if (!Object.prototype.hasOwnProperty.call(pinInfoMap, expected.number)) {
        missingExpectedPins.push({ expected, reason: "PIN_NOT_IN_EXPORTED_NETLIST" });
        continue;
      }
      if (expected.net === null) {
        missingExpectedNets.push({ number: expected.number, reason: "CONTRACT_NET_MISSING" });
        continue;
      }
      const actualPin = pinInfoMap[expected.number];
      const actualNet = actualPin && Object.prototype.hasOwnProperty.call(actualPin, "net")
        ? (actualPin.net === undefined || actualPin.net === null ? null : String(actualPin.net))
        : null;
      const expectedNet = String(expected.net);
      const schematicPin = schematicPinsByNumber.get(expected.number);
      if(!schematicPin || schematicPin.noConnected!==(expectedNet==='')){
        pinNCFlagMismatches.push({number:expected.number,expected:expectedNet==='',actual:schematicPin?.noConnected});
      }
      pinConnectionRows.push({
        number: expected.number,
        name: expected.name || null,
        expectedNet,
        netlistNet: actualNet,
        expectedNoConnect: expectedNet === "",
        schematicNoConnectedFlag: schematicPin ? schematicPin.noConnected : null
      });
      if (actualNet === null || actualNet !== expectedNet) {
        pinNetMismatches.push({ number: expected.number, name: expected.name || null,
          expectedNet, actualNet, expectedNoConnect: expectedNet === "", actualNoConnect: actualNet === "" });
      }
    }
    const duplicateExpectedPinNumbers = [...new Set(expectedPinNumbers.filter((number, index) => expectedPinNumbers.indexOf(number) !== index))];
    const pinSetPass = !!netlistComponent && missingExpectedPins.length === 0 && unexpectedActualPins.length === 0 &&
      duplicateExpectedPinNumbers.length === 0;
    const netsPass = pinSetPass && missingExpectedNets.length === 0 && pinNetMismatches.length === 0 && pinNCFlagMismatches.length===0;
    netAuditByRef.set(ref, {
      status: netsPass ? "PASS_PIN_NETS_AND_NC" : (netlistComponent ? "FAIL_OR_INCOMPLETE_PIN_NETS" : "NETLIST_COMPONENT_NOT_FOUND_OR_DUPLICATE"),
      expectedPinCount: expectedPins.length,
      actualPinCount: actualPinNumbers.length,
      missingExpectedPins,
      unexpectedActualPins,
      duplicateExpectedPinNumbers,
      missingExpectedNets,
      pinNetMismatches,
      pinNCFlagMismatches,
      pinConnectionRows,
      note: "NC要求官方网表空net及实际引脚NoConnected=true；NET脚要求NoConnected=false。"
    });
  }

  const pinStatuses = perComponent.map(component => component.pinAudit.status);
  const pinPassCount = pinStatuses.filter(status => status === "PASS_PIN_NUMBER_SET_MATCH").length;
  const pinMismatchCount = pinStatuses.filter(status => status === "FAIL_PIN_NUMBER_SET_MISMATCH").length;
  const missingPinContractCount = pinStatuses.filter(status => status === "NO_EXPECTED_PINS_IN_CONTRACT").length;
  const missingRefCount = pinStatuses.filter(status => status === "TARGET_REFERENCE_NOT_PRESENT").length;
  const componentSetPass = missingExpectedRefs.length === 0 && extraActualRefs.length === 0 &&
    duplicateExpectedRefs.length === 0 && duplicateActualRefs.length === 0 &&
    unidentifiedActualComponentCount === 0 && invalidExpectedRefCount === 0;
  const allExpectedPinsDefined = components.length > 0 && missingPinContractCount === 0;
  const netAuditStatuses = [...netAuditByRef.values()].map(item => item.status);
  const netPassCount = netAuditStatuses.filter(status => status === "PASS_PIN_NETS_AND_NC").length;
  const netFailOrIncompleteCount = netAuditStatuses.length - netPassCount;
  const netlistParsePass = netlistExport.status === "EXPORTED_AND_PARSED";
  const netlistRefSetPass = !!netlistRefAudit && netlistRefAudit.missingExpectedRefs.length === 0 &&
    netlistRefAudit.extraNetlistRefs.length === 0 && netlistRefAudit.duplicateNetlistRefs.length === 0 &&
    netlistRefAudit.unidentifiedNetlistComponentCount === 0;
  const structuralPass = componentSetPass && allExpectedPinsDefined && pinMismatchCount === 0 && missingRefCount === 0 &&
    netlistParsePass && netlistRefSetPass && netFailOrIncompleteCount === 0;
  return {
    schema: "EASYEDA_READ_ONLY_PIN_ACCEPTANCE_V1",
    status: components.length === 0 ? "CONTRACT_COMPONENTS_REQUIRED" :
      (structuralPass ? "PASS_COMPONENT_SET_PIN_NUMBERS_AND_PIN_NETS" : "FAIL_OR_INCOMPLETE_COMPONENT_SET_PIN_NUMBERS_OR_PIN_NETS"),
    readOnly: true,
    input: { source: cliArgs.contract !== undefined ? "__CLI__.args.contract" : (cliArgs.bom !== undefined ? "__CLI__.args.bom" : "none"),
      contractComponentCount: components.length, contractParseError: rawInput && rawInput.__parseError || null },
    context,
    referenceAudit: {
      expectedComponentCount: components.length,
      actualComponentCount: actualComponents.length,
      missingExpectedRefs,
      extraActualRefs,
      duplicateExpectedRefs,
      duplicateActualRefs,
      unidentifiedActualComponentCount,
      invalidExpectedRefCount
    },
    pinDefinitionAudit: {
      passCount: pinPassCount,
      mismatchCount: pinMismatchCount,
      missingPinContractCount,
      missingReferenceCount: missingRefCount,
      totalExpectedPins: perComponent.reduce((sum, item) => sum + item.expectedPins.length, 0),
      totalActualPins: perComponent.reduce((sum, item) => sum + item.actualPins.length, 0),
      perComponent
    },
    pinNetAudit: {
      status: netlistParsePass ? (netFailOrIncompleteCount === 0 ? "PASS_ALL_PIN_NETS_AND_NC" : "FAIL_OR_INCOMPLETE_PIN_NETS") : "NETLIST_UNAVAILABLE_OR_UNPARSED",
      passCount: netPassCount,
      failOrIncompleteCount: netFailOrIncompleteCount,
      netlistReferenceAudit: netlistRefAudit,
      perComponent: Object.fromEntries(netAuditByRef.entries())
    },
    netlistExport,
    drc: { status: "NOT_RUN", reason: "本阶段不触发DRC；SCH_Drc.check(strict, userInterface, includeVerboseError)的官方调用可单独执行。" },
    notes: [
      "PASS表示合同器件引用集合、实际引脚编号集合以及EasyEDA导出网表中的逐针net/NC一致；不表示原理图DRC或硬件验收通过。",
      "getAllPinsByPrimitiveId用于逐器件读取引脚编号/名称/NC标志；逐针连通网络以官方EasyEDA格式网表pinInfoMap为准。",
      "include_netlist_text=true可将已解析的EasyEDA网表原文一并写入返回结果；默认只返回摘要。",
      "本脚本不调用放置、替换、接线、属性写入、保存或DRC API。"
    ]
  };
})();
