# Pillow Figure 제작 코드

make_figure.py는 기존 Figure의 Pillow 조립 코드를 저장한 것입니다.
기존 DejaVu Serif 글꼴, 배치, 사례와 저장된 panel을 사용합니다.
이미지 재추론, AI 이미지 편집, mask 변경은 하지 않습니다. 기존 PNG/PDF는 변경하지 않았습니다.

실행 명령:

    /home/jovyan/.conda/envs/scale_rae/bin/python /home/jovyan/PGOT/outputs/paper_figures/deor_grounding_2x2_v3/make_figure.py

기본 출력은 이 폴더의 reproduced/입니다. 기존 파일을 덮어쓰지 않습니다.
별도 출력 위치는 --output_dir로 지정할 수 있습니다.

필요 항목: Pillow, 시스템 DejaVu 글꼴, manifest.json에 기록된 원본 panel 및 metadata.
Grounding은 curated_selection.json의 기존 선택 순서 2,6,10,4를 사용합니다.
