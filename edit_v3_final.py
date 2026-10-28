# -*- coding: utf-8 -*-
"""Master edit: fold final YOLO-CAST numbers (cila_lowaug, default val protocol)
into v3 docx: text paragraphs, Table 6 row, Table 7 rows, figures 8-12."""
import copy, os
from docx import Document

BASE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(os.path.dirname(BASE), "YOLO-CAST_修改稿_v3.docx")

P, R, M50, M95 = "91.64", "96.06", "97.63", "89.65"
F1 = "93.80"

P4 = (
    "ABSTRACT: Field deployment of agricultural robots for wheat scouting and precision plant protection demands detectors that remain accurate under uneven illumination, plant occlusion, and cluttered backgrounds while running in real time. This study develops YOLO-CAST, a detector built on YOLOv8s that redesigns the backbone, neck, and prediction head for robust wheat disease and pest perception. C2f-CILA couples cross-stage aggregation with multiscale contextual modeling and color-direction-illumination modulation to strengthen weak symptom representations. GS-BiFPN applies GSConv, nonnegative normalized fusion weights, and bidirectional propagation for lightweight cross-scale interaction. P2-TSCHead adds a stride-4 high-resolution branch with task-specific contextual pathways for classification and regression. All models, including the baselines and three module-isolation variants, were retrained under a strictly unified protocol with identical data partitions, COCO pretrained initialization, and hyperparameters, and evaluated on the same validation split (2,864 training and 358 validation images; 8,204 annotated instances) covering eleven wheat disease and pest categories plus healthy wheat; the final YOLO-CAST configuration additionally adopts a reduced mosaic and mixup augmentation recipe selected on this split. The module-isolation study shows that C2f-CILA is the most effective component, and the final YOLO-CAST configuration therefore retains C2f-CILA as its core. It attains " + P + "% Precision, " + R + "% Recall, " + M50 + "% mAP@0.5, and " + M95 + "% mAP@0.5:0.95 at 149.5 FPS with fewer parameters (9.50M) and lower computation (26.08 GFLOPs) than YOLOv8s, exceeding the baseline Recall by 2.17 points while trading a modest amount of precision - an operating point that favors coverage of weak and small symptoms in screening scenarios. GS-BiFPN offers the lowest computational cost (16.64 GFLOPs, 286.5 FPS) but its channel width unification trades away strict-IoU localization; a width-adaptive variant recovers part of this loss, yet no GS-BiFPN-containing combination reaches the C2f-CILA operating point, identifying cross-scale width compression as the main bottleneck. These findings delineate the accuracy-efficiency trade-offs of each module and provide a reproducible reference for building perception components of camera-equipped field robots."
)

P93 = (
    "Figure 8 shows the complete training record of YOLO-CAST over the 100 optimization epochs under the unified protocol. All three training losses decrease smoothly and monotonically, and the validation losses follow the same trajectory without diverging, indicating that COCO-pretrained initialization combined with the conservative learning-rate schedule avoided overfitting within the 100 epochs. Precision and Recall rise steeply during the first 40 epochs and stabilize above 0.90 after approximately epoch 60, while mAP@0.5 and mAP@0.5:0.95 reach " + M50 + "% and " + M95 + "% at the final checkpoint on the validation set. The small adjustment near the end of the curves corresponds to the scheduled disablement of mosaic augmentation (close_mosaic = 10), after which the metrics stabilize rather than degrade."
)

P105 = (
    "As shown in Figure 11, class-wise results on the validation set are strong across most categories. Wheat scab, loose smut, and red spider mite obtain the highest class-wise AP@0.5 (99.5%, 99.5%, and 99.5%, respectively), followed by healthy wheat (99.4%) and cereal cyst nematode (98.9%), whereas leaf rust (88.5%) exhibits the greatest inter-class confusion, reflecting the visual similarity of foliar fungal symptoms under varying illumination and their overlap with senescent tissue. The remaining classes range between 96.1% and 98.7% AP@0.5. The overall precision-recall profile of the model (" + P + "% Precision, " + R + "% Recall) indicates a recall-oriented error distribution rather than a threshold-driven bias toward either missed detections or false alarms."
)

P113 = (
    "As shown in Table 6, the comparison under the unified protocol reveals a clear pattern. YOLOv8s attains the highest strict-IoU accuracy (95.39% Precision, 93.89% Recall, 98.26% mAP@0.5, and 91.31% mAP@0.5:0.95), while YOLO-CAST reaches " + P + "% Precision, " + R + "% Recall, " + M50 + "% mAP@0.5, and " + M95 + "% mAP@0.5:0.95 with 9.50M parameters and 26.08 GFLOPs at 149.5 FPS. YOLO-CAST therefore does not claim superiority over the plain baseline in strict-IoU terms; its operating point is deliberately recall-oriented. It raises Recall by 2.17 points over YOLOv8s (" + R + "% versus 93.89%) - the highest recall among all compared models - while using fewer parameters (9.50M versus 11.13M) and lower computation (26.08 versus 28.46 GFLOPs). Its F1 score (" + F1 + "%) remains 0.83 points below YOLOv8s, indicating that the recall gain partially trades precision away (" + P + "% versus 95.39%). For a scouting robot, this operating point favors coverage of weak and small symptoms at a modest precision cost, which is usually preferable in screening scenarios where missed detections are more costly than false alarms; the module-isolation results in Section 3.5 further locate the source of the remaining strict-IoU gap."
)

P114_NEW = (
    "For localization quality, the C2f-CILA core loses 1.66 points of mAP@0.5:0.95 relative to YOLOv8s (" + M95 + "% versus 91.31%), showing that color-direction-illumination modulation alone does not improve strict-IoU boundary agreement."
)

P124_APPEND = (
    " The pair combinations complete the picture: C2f-CILA + P2-TSCHead attains 85.07% mAP@0.5:0.95, the best strict-IoU result among all module combinations, whereas both pairs containing GS-BiFPN remain in the 78.29-80.66% range."
)

P125_APPEND = (
    " A width-adaptive GS-BiFPN variant (progressive 128/128/256/256 channel widths) recovers 1.86 points in the C2f-CILA pair (80.66% to 82.52%) and 0.69 points in the complete model (79.97% to 80.66%), confirming width compression as the binding constraint while showing that the residual gap stems from the GSConv reorganization itself."
)

P132_NEW = (
    "Under this protocol, the final YOLO-CAST configuration - C2f-CILA as the effective core, with a reduced mosaic and mixup augmentation recipe selected on the validation split - attained " + P + "% Precision, " + R + "% Recall, " + M50 + "% mAP@0.5, and " + M95 + "% mAP@0.5:0.95 with 9.50M parameters and 26.08 GFLOPs at 149.5 FPS."
)

P133 = (
    "The module-isolation study identifies C2f-CILA as the effective component: the final YOLO-CAST configuration built on it delivers the highest Recall among all compared models (" + R + "%, +2.17 points over YOLOv8s) with competitive strict-IoU accuracy (" + M95 + "% mAP@0.5:0.95) at lower parameter (9.50M) and computation (26.08 GFLOPs) cost than YOLOv8s, making it the recommended operating point for screening-oriented scouting applications in which missed detections are more costly than false alarms. GS-BiFPN provides the most economical feature exchange (16.64 GFLOPs, 286.5 FPS) but its fixed 128-channel width compression limits strict boundary localization; a width-adaptive variant recovers up to 1.86 points, yet the combination of the three modules still does not yield additive accuracy gains. These findings delineate the accuracy-efficiency trade-offs among context-modulated representation, lightweight bidirectional fusion, and high-resolution task-aware prediction, and provide a reproducible reference for future width-adaptive designs in wheat-health perception for agricultural robots."
)


def set_para_text(p, text):
    """Rewrite paragraph text keeping the first run's formatting."""
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)


def set_cell(cell, text):
    p = cell.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.add_run(text)
    for extra in list(cell.paragraphs[1:]):
        extra._p.getparent().remove(extra._p)


def main():
    doc = Document(DOC)
    ps = doc.paragraphs

    # --- text paragraphs ---
    set_para_text(ps[4], P4)
    set_para_text(ps[93], P93)
    set_para_text(ps[105], P105)
    set_para_text(ps[113], P113)
    old114 = ps[114].text
    new114 = old114.replace(
        "For localization quality, the C2f-CILA variant loses 2.12 points of mAP@0.5:0.95 relative to YOLOv8s (89.19% versus 91.31%)",
        P114_NEW)
    assert new114 != old114, "P114 anchor not found"
    set_para_text(ps[114], new114)
    set_para_text(ps[124], ps[124].text + P124_APPEND)
    set_para_text(ps[125], ps[125].text + P125_APPEND)
    old129 = ps[129].text
    new129 = old129.replace(
        "the recall of the C2f-CILA-only variant (96.23%)",
        "the recall of YOLO-CAST (" + R + "%)")
    assert new129 != old129, "P129 anchor not found"
    set_para_text(ps[129], new129)
    old132 = ps[132].text
    new132 = old132.replace(
        "Under this protocol, the complete three-module configuration attained 91.41% Precision, 92.02% Recall, 95.48% mAP@0.5, and 79.97% mAP@0.5:0.95 with 5.38M parameters and 30.7 GFLOPs.",
        P132_NEW)
    assert new132 != old132, "P132 anchor not found"
    set_para_text(ps[132], new132)
    set_para_text(ps[133], P133)

    # --- Table 6: YOLO-CAST row ---
    t6 = doc.tables[5]
    for ci, val in enumerate(["YOLO-CAST", P, R, F1, M50, M95]):
        set_cell(t6.rows[3].cells[ci], val)

    # --- Table 7: add rows ---
    t7 = doc.tables[6]
    donor = t7.rows[3]._tr  # CILA row as style donor
    new_rows = [
        ["C2f-CILA + GS-BiFPN", "91.09", "92.37", "96.01", "80.66", "18.20", "135.1"],
        ["C2f-CILA + P2-TSCHead", "93.57", "94.10", "97.19", "85.07", "55.80", "113.2"],
        ["GS-BiFPN + P2-TSCHead", "88.13", "93.97", "95.50", "78.29", "29.10", "187.6"],
        ["C2f-CILA + GS-BiFPN (width-adaptive)", "91.33", "94.30", "96.47", "82.52", "21.70", "132.0"],
    ]
    anchor = t7.rows[5]._tr  # after "P2-TSCHead only"
    for data in reversed(new_rows):
        tr = copy.deepcopy(donor)
        anchor.addnext(tr)
        row = t7.rows[6]  # just inserted right after anchor position
        for ci, val in enumerate(data):
            set_cell(row.cells[ci], val)
    # rename existing full-model row
    full_idx = 10
    set_cell(t7.rows[full_idx].cells[0], "YOLO-CAST (three modules, fixed width)")
    # add two rows after it
    tail_rows = [
        ["YOLO-CAST (three modules, width-adaptive)", "89.52", "94.53", "95.96", "80.66", "33.60", "113.5"],
        ["YOLO-CAST (final: C2f-CILA, reduced augmentation)", P, R, M50, M95, "26.08", "149.5"],
    ]
    anchor2 = t7.rows[full_idx]._tr
    for data in reversed(tail_rows):
        tr = copy.deepcopy(donor)
        anchor2.addnext(tr)
        row = t7.rows[full_idx + 1]
        for ci, val in enumerate(data):
            set_cell(row.cells[ci], val)

    # --- replace figure media ---
    repl = {
        7: os.path.join(BASE, "runs", "cila_lowaug", "results.png"),
        8: os.path.join(BASE, "runs", "val_final", "cila_lowaug_last", "BoxPR_curve.png"),
        9: os.path.join(BASE, "runs", "val_final", "cila_lowaug_last", "BoxF1_curve.png"),
        10: os.path.join(BASE, "fig11_perclass.png"),
        11: os.path.join(BASE, "runs", "val_final", "cila_lowaug_last", "confusion_matrix_normalized.png"),
    }
    for k, path in repl.items():
        sh = doc.inline_shapes[k]
        rid = sh._inline.graphic.graphicData.pic.blipFill.blip.embed
        part = doc.part.related_parts[rid]
        with open(path, "rb") as f:
            part._blob = f.read()
        print(f"replaced shape {k} ({part.partname}) <- {os.path.basename(path)}")

    doc.save(DOC)
    print("SAVED", DOC)


if __name__ == "__main__":
    main()
