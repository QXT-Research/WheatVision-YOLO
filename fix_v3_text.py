# -*- coding: utf-8 -*-
"""YOLO-CAST v3 批量修复：重复段落、乱码、行号残留、F_loc 解释。
从文档后部往前执行 replace，最后做插入与文本级 find_and_replace。
"""
import json
import subprocess
import sys

EDSDK_DIR = r"G:/安装的软件/workbuddy/resources/app.asar.unpacked/resources/plugins/workbuddy-builtin/skills/workbuddy-local-office-edit"
FILE_ID = "C:/Users/Administrator/WorkBuddy/小麦yolo/YOLO-CAST_修改稿_v3.docx"


def call(tool, args):
    args = dict(args)
    args.setdefault("file_id", FILE_ID)
    r = subprocess.run(
        [sys.executable, "edsdk.py", "call", tool, "--json", json.dumps(args, ensure_ascii=False)],
        cwd=EDSDK_DIR, capture_output=True, text=True, encoding="utf-8", timeout=180)
    out = r.stdout.strip()
    if r.returncode != 0 or out.startswith("error"):
        raise RuntimeError(f"{tool} failed: {out[:500]} | {r.stderr[:300]}")
    return json.loads(out) if out.startswith("{") else out


def structure():
    r = subprocess.run(
        [sys.executable, "edsdk.py", "call", "doc_resolve_document_structure",
         "--json", json.dumps({"limit": 0, "text_preview_length": 60, "file_id": FILE_ID}, ensure_ascii=False)],
        cwd=EDSDK_DIR, capture_output=True, text=True, encoding="utf-8", timeout=300)
    if not r.stdout.strip().startswith("{"):
        raise RuntimeError(f"structure() raw stdout={r.stdout[:400]!r} stderr={r.stderr[:400]!r}")
    return json.loads(r.stdout)["nodes"]


def find_node(nodes, key):
    for n in nodes:
        if key in n.get("text_preview", ""):
            return n
    raise KeyError(f"node not found: {key}")


def replace_node(node, text):
    res = call("doc_replace_text", {
        "ranges": [{"begin": node["start_index"], "end": node["end_index"] - 1}],
        "text": text})
    print("  replaced:", node["text_preview"][:40], "->", res.get("message", "ok"))


TEXTS = {
    "Conflicts of Interest": "Conflicts of Interest: The author states that there is no financial or non-financial conflict of interest that could affect the study and the interpretation of its results.",
    "Ethics Approval": "Ethics Approval: Not applicable.",
    "To verify the effectiveness": "To verify the effectiveness of the proposed components, controlled experiments were designed: three YOLOv8s variants containing C2f-CILA, GS-BiFPN, or P2-TSCHead individually, and the complete model containing all three modules. Every experiment used the same dataset split, pretrained initialization, input size, number of epochs, hyperparameters, and hardware environment. Comparing the single-module variants and their pairwise combinations with the complete model reveals both the independent contribution of each component and their interaction effects.",
    "For the three related methods": "For the three related methods, F1 was calculated from the Precision and Recall reported in the original article as 2PR/(P + R); a dash would indicate a metric not reported in the corresponding source. Because task classes, data distributions, training strategies, and timing hardware differ, literature-reported values are not used to calculate improvements of the present model and do not constitute evidence of superiority under a unified protocol. This separation retains the technical reference supplied by recent wheat-detection studies while avoiding the misrepresentation of heterogeneous experimental results as direct reproductions.",
    "Equations (8)": "Equations (8)-(11) form a continuous computational chain comprising low-cost transformation, scale-contribution estimation, fixed-width node fusion, and bidirectional semantic exchange. The outputs {B2, B3, B4, B5} share a unified detection-channel width and are passed directly to P2-TSCHead. B2 preserves abundant small-lesion and insect details, whereas B5 carries stronger category-level semantics.",
    "In Eq. (10), Ai": "In Eq. (10), Ai\u2192o comprises the necessary 1 \u00d7 1 projection and upsampling or downsampling operation.",
    "Because incoming features": "Because incoming features may differ in resolution and channel count, they cannot be added directly. Let Fi be the ith input to node o, originating from a lateral connection, a top-down upsampled feature, or a bottom-up downsampled feature. The alignment operator Ai\u2192o matches its spatial resolution and channel width before normalized weighted fusion:",
    "In Eq. (9), wi": "In Eq. (9), wi is learned by backpropagation, \u0175i is the nonnegative normalized weight of the ith input, n is the number of inputs to the current node, and \u03b5 prevents division by zero. The weights approximately sum to one, making the contributions of different scales comparable.",
    "GS-BiFPN follows four": "GS-BiFPN follows four steps: lightweight channel reconstruction, normalized fusion weighting, node-wise feature refinement, and bidirectional propagation. Its purpose is to exchange deep semantics and shallow details under a fixed channel width while avoiding the redundant convolutions and channel expansion associated with direct concatenation. Let Pl \u2208 \u211d(B\u00d7Cl\u00d7Hl\u00d7Wl), where l \u2208 {2, 3, 4, 5} denotes the feature level. These inputs originate from C2f-CILA or from aligned features within the fusion pathway. Figure 5 shows the overall structure.",
}

FLOC_TEXT = ("In Eq. (2), F_loc denotes the local texture branch extracted from the shared "
             "representation Z by a 3 \u00d7 3 depthwise convolution, whereas F5, F7 and F11 are "
             "multi-scale context branches obtained by 5 \u00d7 5, 7 \u00d7 7 and 11 \u00d7 11 depthwise "
             "convolutions; the two large kernels are decomposed into separable k \u00d7 1 and 1 \u00d7 k "
             "forms to keep the parameter budget low. Because lesion appearance varies from fine "
             "speckles to spreading patches, these branches cover complementary receptive fields at "
             "negligible additional cost, and F_loc is reused as the local pathway in the response "
             "modulation of Eqs. (4)-(7).")

REF_FIXES = [
    ["Access, 5 20", "Access, 20"],
    ["Based on 5 Improved YOLOv8 Model", "Based on Improved YOLOv8 Model"],
    ["16(3). 5 [CrossRef]", "16(3). [CrossRef]"],
    [". 5 Transactions", ". Transactions"],
    ["Pests. 5 Agriculture", "Pests. Agriculture"],
    ["YOLOv5. 5 Remote Sensing", "YOLOv5. Remote Sensing"],
    ["YOLOv10. 5 Agronomy", "YOLOv10. Agronomy"],
    ["Sciences, 5 2024", "Sciences, 2024"],
    ["UAV 5 Remote Sensing Images", "UAV Remote Sensing Images"],
    ["Fusarium 5 Head Blight", "Fusarium Head Blight"],
    ["109552. 5 [CrossRef]", "109552. [CrossRef]"],
    ["Spikes 5 in Natural", "Spikes in Natural"],
    ["Based on 5 RGB Images", "Based on RGB Images"],
    ["Network 5 Based on YOLOv7", "Network Based on YOLOv7"],
    ["Airborne 5 Fungal", "Airborne Fungal"],
    ["on an 5 Improved YOLOv8n", "on an Improved YOLOv8n"],
    ["Multi-Stage 5 Attention", "Multi-Stage Attention"],
    ["Improved 5 YOLOv5s", "Improved YOLOv5s"],
    ["Multi-Scale 5 Information", "Multi-Scale Information"],
    ["Disease 5 Recognition", "Disease Recognition"],
    ["OSWSDet) 5 in UAV", "OSWSDet) in UAV"],
]


def main():
    nodes = structure()
    # 1) descending-order full-paragraph replacements / delete
    order = ["Conflicts of Interest", "Ethics Approval", "The predictions keep continuous",
             "To verify the effectiveness", "For the three related methods", "Equations (8)",
             "In Eq. (10), Ai", "Because incoming features", "In Eq. (9), wi",
             "GS-BiFPN follows four"]
    for key in order:
        n = find_node(nodes, key)
        if key == "The predictions keep continuous":
            res = call("doc_delete_paragraph", {"idx": n["start_index"]})
            print("  deleted stray paragraph @", n["start_index"], res.get("message", ""))
        else:
            replace_node(n, TEXTS[key])

    # 2) abstract dedupe
    nodes = structure()
    ab = find_node(nodes, "ABSTRACT: ABSTRACT:")
    r = subprocess.run(
        [sys.executable, "-c",
         "import json,subprocess,sys;"
         "from docx import Document;d=Document(r'C:/Users/Administrator/WorkBuddy/小麦yolo/YOLO-CAST_修改稿_v3.docx');"
         "t=d.paragraphs[4].text;"
         "i=t.find('ABSTRACT: Field deployment');"
         "clean='ABSTRACT: '+t[t.find('Field deployment', i):i].strip() if i>0 else t;"
         "print(json.dumps(clean))"],
        capture_output=True, text=True, encoding="utf-8")
    clean_abs = json.loads(r.stdout.strip())
    print("abstract: old len", ab["end_index"] - ab["start_index"], "new len", len(clean_abs))
    replace_node(ab, clean_abs)

    # 3) insert F_loc explanation before 'The scale-selective weights'
    nodes = structure()
    n = find_node(nodes, "The scale-selective weights")
    res = call("doc_insert_paragraph_with_text", {"idx": n["start_index"], "text": FLOC_TEXT})
    print("  inserted F_loc paragraph:", str(res)[:120])

    # 4) reference stray-5 fixes (text-level, coordinate-free)
    for old, new in REF_FIXES:
        res = call("doc_find_and_replace", {"text": old, "replacement": new})
        n_rep = res.get("replaced_count", res.get("count", "?")) if isinstance(res, dict) else "?"
        print(f"  ref fix: {old[:28]!r} -> {n_rep}")

    # 5) save
    res = call("save_file", {})
    print("SAVED:", str(res)[:200])


if __name__ == "__main__":
    main()
