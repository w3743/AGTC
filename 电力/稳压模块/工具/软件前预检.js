// 嘉立创EDA官方API只读预检。代码由API Bridge在EDA运行时执行；不访问文件系统、fetch或浏览器DOM。
// 运行参数：__CLI__.args.contract，或 __CLI__.args.bom（均接受参数对象或JSON字符串）。
return await (async function runReadOnlyPreflight() {
  const DEFAULT_PAGE_UUID = "9546dca0bc8bce7f";
  const PAGE_SIZE = 100;
  const MAX_SEARCH_PAGES = 10;
  const cliArgs = typeof __CLI__ !== "undefined" && __CLI__ && __CLI__.args && typeof __CLI__.args === "object"
    ? __CLI__.args
    : {};

  function parseObject(value) {
    if (typeof value !== "string") return value;
    try { return JSON.parse(value); } catch (error) {
      return { __parseError: String(error && error.message || error) };
    }
  }

  const rawInput = parseObject(cliArgs.contract !== undefined
    ? cliArgs.contract
    : (cliArgs.bom !== undefined ? cliArgs.bom : {}));
  const contract = rawInput && rawInput.contract ? rawInput.contract : rawInput;
  const components = Array.isArray(contract && contract.components) ? contract.components : [];
  const targetPageUuid = String(
    (contract && (contract.target_page_uuid || contract.targetPageUuid)) ||
    (rawInput && (rawInput.target_page_uuid || rawInput.targetPageUuid)) ||
    cliArgs.target_page_uuid || cliArgs.targetPageUuid || DEFAULT_PAGE_UUID
  );
  const expectedProjectUuid = (contract && (contract.target_project_uuid || contract.expectedProjectUuid || contract.project_uuid)) ||
    (rawInput && (rawInput.target_project_uuid || rawInput.expectedProjectUuid || rawInput.project_uuid)) || null;
  const libraryUuid = (contract && (contract.library_uuid || contract.libraryUuid)) || (rawInput && (rawInput.library_uuid || rawInput.libraryUuid)) || undefined;

  function firstDefined(object, snake, camel) {
    if (!object) return undefined;
    return object[snake] !== undefined ? object[snake] : object[camel];
  }
  function norm(value) {
    return value === undefined || value === null ? "" : String(value).trim().toUpperCase();
  }
  function safeState(object, method) {
    try { return object && typeof object[method] === "function" ? object[method]() : undefined; }
    catch (error) { return { __readError: String(error && error.message || error) }; }
  }
  function asExpectedPin(pin) {
    return {
      number: pin && pin.number !== undefined && pin.number !== null ? String(pin.number).trim() : "",
      name: pin && pin.name !== undefined && pin.name !== null ? String(pin.name).trim() : ""
    };
  }
  function comparePinDefinitions(expectedPins, actualPins) {
    if (!Array.isArray(expectedPins) || expectedPins.length === 0) {
      return { status: "NO_EXPECTED_PINS_IN_CONTRACT", expectedCount: 0, actualCount: actualPins.length,
        missingExpected: [], unexpectedActual: actualPins, nameMismatches: [], duplicateExpectedNumbers: [], duplicateActualNumbers: [] };
    }
    const expected = expectedPins.map(asExpectedPin);
    const expectedNumbers = expected.map(pin => pin.number).filter(Boolean);
    const actualNumbers = actualPins.map(pin => pin.number).filter(Boolean);
    const duplicates = values => [...new Set(values.filter((value, index) => values.indexOf(value) !== index))];
    const expectedByNumber = new Map(expected.filter(pin => pin.number).map(pin => [pin.number, pin]));
    const actualByNumber = new Map(actualPins.filter(pin => pin.number).map(pin => [pin.number, pin]));
    const missingExpected = expected.filter(pin => pin.number
      ? !actualByNumber.has(pin.number)
      : (pin.name && !actualPins.some(actual => actual.name === pin.name)));
    const unexpectedActual = actualPins.filter(pin => pin.number
      ? !expectedByNumber.has(pin.number)
      : !expected.some(want => want.name && want.name === pin.name));
    const nameMismatches = expected.filter(want => want.number && want.name && actualByNumber.has(want.number))
      .map(want => ({ expected: want, actual: actualByNumber.get(want.number) }))
      .filter(pair => pair.actual.name !== pair.expected.name);
    const duplicateExpectedNumbers = duplicates(expectedNumbers);
    const duplicateActualNumbers = duplicates(actualNumbers);
    const passed = missingExpected.length === 0 && unexpectedActual.length === 0 &&
      duplicateExpectedNumbers.length === 0 && duplicateActualNumbers.length === 0;
    return { status: passed ? "PASS_PIN_NUMBER_SET_MATCH" : "FAIL_PIN_NUMBER_SET_MISMATCH",
      expectedCount: expected.length, actualCount: actualPins.length, missingExpected, unexpectedActual,
      nameMismatches, duplicateExpectedNumbers, duplicateActualNumbers,
      note: "引脚名称仅供人工检查；符号名可能与BOM名不同。" };
  }
  function componentRef(item) { return String(firstDefined(item, "ref", "reference") || "").trim(); }
  function targetMpn(item) { return String(firstDefined(item, "manufacturer_part", "manufacturerPart") || "").trim(); }
  function targetC(item) { return String(firstDefined(item, "supplier_part", "supplierPart") || "").trim(); }
  function targetLibraryUuid(item) { return firstDefined(item, "library_uuid", "libraryUuid") || libraryUuid; }

  const project = await eda.dmt_Project.getCurrentProjectInfo();
  const currentDocument = await eda.dmt_SelectControl.getCurrentDocumentInfo();
  const currentPage = await eda.dmt_Schematic.getCurrentSchematicPageInfo();
  const pages = await eda.dmt_Schematic.getAllSchematicPagesInfo();
  const pageListed = Array.isArray(pages) && pages.some(page => page && page.uuid === targetPageUuid);
  const documentIsSchematicPage = !!currentDocument && currentDocument.documentType === 1;
  const pageMatches = !!currentPage && currentPage.uuid === targetPageUuid &&
    !!currentDocument && currentDocument.uuid === targetPageUuid;
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
      schema: "EASYEDA_READ_ONLY_PREFLIGHT_V1",
      status: "BLOCKED_WRONG_OR_UNAVAILABLE_PROJECT_PAGE",
      readOnly: true,
      input: { source: cliArgs.contract !== undefined ? "__CLI__.args.contract" : (cliArgs.bom !== undefined ? "__CLI__.args.bom" : "none"),
        contractComponentCount: components.length, contractParseError: rawInput && rawInput.__parseError || null },
      context,
      inventory: null,
      referenceAudit: null,
      libraryLookup: null,
      notes: ["仅在目标工程与原理图页处于当前焦点时继续读取；本次没有读取或改变任何器件。"]
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
    actualComponents.push({
      ref: ref || null,
      primitiveId: primitiveId && !primitiveId.__readError ? primitiveId : null,
      name: safeState(primitive, "getState_Name") || null,
      manufacturerPart: safeState(primitive, "getState_ManufacturerId") || null,
      supplierPart: safeState(primitive, "getState_SupplierId") || null,
      componentLibraryUuid: device && device.libraryUuid || null,
      componentUuid: device && device.uuid || null,
      pins,
      pinsRead: Array.isArray(pinsRead)
    });
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
  const missingExpectedRefs = [...expectedRefSet].filter(ref => !actualRefSet.has(ref));
  const extraActualRefs = [...actualRefSet].filter(ref => !expectedRefSet.has(ref));

  const componentAudit = components.map(target => {
    const ref = componentRef(target);
    const actualRows = actualByRef.get(ref) || [];
    const actual = actualRows.length === 1 ? actualRows[0] : null;
    const pins = Array.isArray(target.pins) ? target.pins : [];
    const pinAudit = actual ? comparePinDefinitions(pins, actual.pins) : {
      status: actualRows.length > 1 ? "AMBIGUOUS_DUPLICATE_ACTUAL_REFERENCE" : "TARGET_REFERENCE_NOT_PRESENT",
      expectedCount: pins.length, actualCount: null, missingExpected: pins, unexpectedActual: [], nameMismatches: [],
      duplicateExpectedNumbers: [], duplicateActualNumbers: []
    };
    const mpn = targetMpn(target);
    const cNumber = targetC(target);
    const actualMpn = actual && actual.manufacturerPart || "";
    const actualC = actual && actual.supplierPart || "";
    const identityMismatch = !actual || (!!mpn && norm(actualMpn) !== norm(mpn)) || (!!cNumber && norm(actualC) !== norm(cNumber));
    return {
      ref,
      manufacturerPart: mpn || null,
      supplierPart: cNumber || null,
      scope: target.scope || null,
      footprint: target.footprint || null,
      actualCount: actualRows.length,
      actual: actual ? { manufacturerPart: actualMpn || null, supplierPart: actualC || null,
        componentLibraryUuid: actual.componentLibraryUuid, componentUuid: actual.componentUuid } : null,
      expectedPins: pins,
      pinAudit,
      libraryLookupRequired: identityMismatch,
      libraryLookupReason: !actual ? "REFERENCE_NOT_IN_CURRENT_PAGE" : (identityMismatch ? "TARGET_PART_DIFFERS_FROM_CURRENT_INSTANCE" : "INSTANCE_PART_FIELDS_MATCH_TARGET")
    };
  });

  const queryGroups = new Map();
  for (const target of components) {
    const ref = componentRef(target);
    const rows = actualByRef.get(ref) || [];
    const actual = rows.length === 1 ? rows[0] : null;
    const mpn = targetMpn(target);
    const cNumber = targetC(target);
    if (!mpn && !cNumber) continue;
    const identityMismatch = !actual || (!!mpn && norm(actual.manufacturerPart) !== norm(mpn)) ||
      (!!cNumber && norm(actual.supplierPart) !== norm(cNumber));
    if (!identityMismatch) continue;
    const key = [targetLibraryUuid(target) || "<system>", norm(cNumber), norm(mpn)].join("|");
    if (!queryGroups.has(key)) queryGroups.set(key, { libraryUuid: targetLibraryUuid(target), supplierPart: cNumber, manufacturerPart: mpn, refs: [] });
    if (ref) queryGroups.get(key).refs.push(ref);
  }

  function identityKey(item) { return String(item && item.libraryUuid || "") + "|" + String(item && item.uuid || ""); }
  function matchingMpnEvidence(item, partNumber) {
    const wanted = norm(partNumber);
    const evidence = [];
    const add = (source, key, value) => {
      if (typeof value === "string" || typeof value === "number") {
        if (norm(value) === wanted) evidence.push({ source, key, value: String(value) });
      }
    };
    if (!item || !wanted) return evidence;
    add("name", "name", item.name);
    add("manufacturerId", "manufacturerId", item.manufacturerId);
    const visit = (obj, prefix) => {
      if (!obj || typeof obj !== "object") return;
      for (const [key, value] of Object.entries(obj)) {
        const fullKey = prefix ? prefix + "." + key : key;
        if (/manufacturer.?part|part.?number|\bmpn\b|model/i.test(key)) add("otherProperty", fullKey, value);
        if (value && typeof value === "object") visit(value, fullKey);
      }
    };
    visit(item.otherProperty || item.property, "");
    return evidence;
  }
  function conciseCandidate(item, detail, target, mpnSearchItem) {
    const evidence = matchingMpnEvidence(item, target.manufacturerPart);
    const searchEvidence = matchingMpnEvidence(mpnSearchItem, target.manufacturerPart);
    const detailEvidence = matchingMpnEvidence(detail, target.manufacturerPart);
    const combinedEvidence = evidence.concat(searchEvidence, detailEvidence);
    const source = item && item.otherProperty || {};
    const detailProperty = detail && detail.property || {};
    return {
      name: item && item.name || detail && detail.name || null,
      uuid: item && item.uuid || detail && detail.uuid || null,
      libraryUuid: item && item.libraryUuid || detail && detail.libraryUuid || null,
      symbol: item && item.symbol || null,
      footprint: item && item.footprint || null,
      supplierId: item && (item.supplierId || source["Supplier Part"] || source["Supplier Part ID"]) || null,
      manufacturerId: item && (item.manufacturerId || source["Manufacturer Part"] || source["Manufacturer Part Number"]) || null,
      mpnExact: combinedEvidence.length > 0,
      mpnEvidence: combinedEvidence,
      detailRead: !!detail,
      detailPropertyKeys: detailProperty && typeof detailProperty === "object" ? Object.keys(detailProperty) : []
    };
  }

  const libraryLookup = [];
  for (const group of queryGroups.values()) {
    const target = { manufacturerPart: group.manufacturerPart, supplierPart: group.supplierPart };
    const candidateMap = new Map();
    let lcscCandidates = [];
    let lcscError = null;
    if (group.supplierPart) {
      try {
        // 数组重载 + allowMultiMatch=true，保留同C编号的全部库项，不取首项。
        lcscCandidates = await eda.lib_Device.getByLcscIds([group.supplierPart], group.libraryUuid, true) || [];
      } catch (error) { lcscError = String(error && error.message || error); }
      for (const item of lcscCandidates) {
        const key = identityKey(item);
        if (key !== "|") candidateMap.set(key, { item, mpnSearchItem: null, fromLcscLookup: true, fromMpnSearch: false });
      }
    }

    let exactMpnSearchItems = [];
    let searchError = null;
    let searchPagesRead = 0;
    let searchLimitReached = false;
    if (group.manufacturerPart) {
      try {
        for (let page = 1; page <= MAX_SEARCH_PAGES; page += 1) {
          const resultPage = await eda.lib_Device.search(group.manufacturerPart, group.libraryUuid, undefined, undefined, PAGE_SIZE, page) || [];
          searchPagesRead = page;
          for (const item of resultPage) {
            if (matchingMpnEvidence(item, group.manufacturerPart).length === 0) continue;
            exactMpnSearchItems.push(item);
            const key = identityKey(item);
            if (key === "|") continue;
            if (!candidateMap.has(key)) candidateMap.set(key, { item, mpnSearchItem: item, fromLcscLookup: false, fromMpnSearch: true });
            else {
              candidateMap.get(key).fromMpnSearch = true;
              candidateMap.get(key).mpnSearchItem = item;
            }
          }
          if (resultPage.length < PAGE_SIZE) break;
          if (page === MAX_SEARCH_PAGES) searchLimitReached = true;
        }
      } catch (error) { searchError = String(error && error.message || error); }
    }

    const candidates = [];
    for (const entry of candidateMap.values()) {
      let detail;
      try {
        if (entry.item && entry.item.uuid) detail = await eda.lib_Device.get(entry.item.uuid, entry.item.libraryUuid || group.libraryUuid);
      } catch (error) { entry.detailError = String(error && error.message || error); }
      const summary = conciseCandidate(entry.item, detail, target, entry.mpnSearchItem);
      const cMatch = !group.supplierPart || entry.fromLcscLookup || norm(summary.supplierId) === norm(group.supplierPart);
      const exactMatch = summary.mpnExact && cMatch;
      candidates.push({ ...summary, fromExactLcscLookup: entry.fromLcscLookup, fromExactMpnSearch: entry.fromMpnSearch,
        supplierPartMatchesTarget: cMatch, exactTargetMatch: exactMatch, detailError: entry.detailError || null });
    }
    const exactCandidates = candidates.filter(candidate => candidate.exactTargetMatch);
    let status;
    if (!group.manufacturerPart && group.supplierPart) status = "C_FOUND_MPN_UNVERIFIED";
    else if (!group.supplierPart && group.manufacturerPart) {
      if (searchLimitReached || searchError) status = "MPN_SEARCH_INCOMPLETE_NO_SELECTION";
      else status = exactCandidates.length === 1 ? "EXACT_MPN_ONLY_C_UNVERIFIED" : (exactCandidates.length > 1 ? "AMBIGUOUS_MPN_ONLY_NO_SELECTION" : "EXACT_MPN_NOT_FOUND_OR_UNVERIFIED");
    }
    else status = exactCandidates.length === 1 ? "UNIQUE_EXACT_MPN_AND_C_MATCH" : (exactCandidates.length > 1 ? "AMBIGUOUS_EXACT_MATCHES_NO_SELECTION" : "NO_VERIFIED_EXACT_MATCH");
    libraryLookup.push({
      refs: group.refs,
      manufacturerPart: group.manufacturerPart || null,
      supplierPart: group.supplierPart || null,
      libraryUuid: group.libraryUuid || null,
      status,
      uniqueCandidate: status === "UNIQUE_EXACT_MPN_AND_C_MATCH" ? exactCandidates[0] : null,
      exactCandidateCount: exactCandidates.length,
      candidates,
      lcscLookupError: lcscError,
      mpnSearchError: searchError,
      mpnSearchPagesRead: searchPagesRead,
      mpnSearchLimitReached: searchLimitReached,
      exactMpnSearchResultCount: exactMpnSearchItems.length
    });
  }

  const allPinAudits = componentAudit.map(item => item.pinAudit.status);
  const missingContractComponents = components.length === 0;
  return {
    schema: "EASYEDA_READ_ONLY_PREFLIGHT_V1",
    status: missingContractComponents ? "CONTRACT_COMPONENTS_REQUIRED" : "PREFLIGHT_READ_COMPLETE_REVIEW_FLAGS_BELOW",
    readOnly: true,
    input: {
      source: cliArgs.contract !== undefined ? "__CLI__.args.contract" : (cliArgs.bom !== undefined ? "__CLI__.args.bom" : "none"),
      contractComponentCount: components.length,
      defaultTargetPageUuidUsed: !(contract && (contract.target_page_uuid || contract.targetPageUuid)) &&
        !(rawInput && (rawInput.target_page_uuid || rawInput.targetPageUuid)) && !cliArgs.target_page_uuid && !cliArgs.targetPageUuid,
      contractParseError: rawInput && rawInput.__parseError || null
    },
    context,
    inventory: {
      actualComponentCount: actualComponents.length,
      actualComponents,
      unidentifiedComponents: actualComponents.filter(item => !item.ref),
      componentsWithPinReadFailure: actualComponents.filter(item => !item.pinsRead).map(item => item.ref)
    },
    referenceAudit: {
      expectedComponentCount: components.length,
      missingExpectedRefs,
      extraActualRefs,
      duplicateExpectedRefs,
      duplicateActualRefs,
      perComponent: componentAudit,
      pinDefinitionSummary: {
        pass: allPinAudits.filter(status => status === "PASS_PIN_NUMBER_SET_MATCH").length,
        mismatch: allPinAudits.filter(status => status === "FAIL_PIN_NUMBER_SET_MISMATCH").length,
        noExpectedPinContract: allPinAudits.filter(status => status === "NO_EXPECTED_PINS_IN_CONTRACT").length,
        notPresent: allPinAudits.filter(status => status === "TARGET_REFERENCE_NOT_PRESENT").length
      }
    },
    libraryLookup,
    notes: [
      "庫查詢只對不存在或料號欄位不符合目標的位號執行；按精確C編號與MPN查詢，不自动选用任何结果。",
      "getAllPinsByPrimitiveId读取引脚编号/名称，不包含连通网络；逐针网络归属需由独立网表阶段或外部连接合同验收。",
      "本脚本只读；未调用器件放置、替换、接线、属性写入、保存或DRC API。"
    ]
  };
})();
