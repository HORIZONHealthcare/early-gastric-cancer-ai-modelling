# Data access

No patient-level data or endoscopic images are distributed with this repository.

The original study data are controlled clinical records. De-identified data may be available upon reasonable request and subject to institutional approval. Requests should be directed to Yi Lu at [luyi45@mail.sysu.edu.cn](mailto:luyi45@mail.sysu.edu.cn).

## Data roles

The modelling workflow uses a controlled internal development dataset. Logistic-regression and LightGBM models additionally use a separate external dataset for validation. Task-specific deep-learning and VLM experiments use prespecified image-available subsets of the internal dataset.

The public code requests input paths at runtime and does not contain local absolute paths or original data filenames. Because the source datasets are not publicly distributed, their original workbook names, worksheet names, and field dictionary are not reproduced in this documentation. Researchers adapting the framework should map their own governed dataset to the variables referenced by the relevant script and verify all task definitions before analysis.

## Endoscopic-image layout

Image-based scripts request an image-root directory. In the original workflow, the subdirectory name matches the value used by the script's `NAME_COLUMN`, and each subdirectory contains one or more supported image files:

```text
image-root/
├── <sample-directory-1>/
│   ├── image-01.jpg
│   └── image-02.jpg
└── <sample-directory-2>/
    └── image-01.png
```

For a new cohort, use de-identified sample identifiers and adapt the image-lookup logic in a private working copy. Never commit the mapping between sample identifiers and patient identities.

## Adapting another dataset

1. Create a private copy of the relevant script.
2. Map your cohort's variables to the script constants or preparation functions.
3. Confirm units and categorical encodings before fitting a model.
4. Keep source data and generated patient-level outputs outside version control.

The `.gitignore` excludes common spreadsheet, image-output, checkpoint, credential, and results locations, but repository users remain responsible for checking staged files before every commit.
