# Khởi động literature review và kiểm tra skill/tool — 21/10/2026

> **HISTORICAL_SNAPSHOT — giữ provenance của lượt 21/10.** Đã thêm chỉ dẫn
> đồng bộ ngày 22/10/2026; các bảng tool/skill/versions, số lượng thư mục và
> kết quả search bên dưới vẫn mô tả thời điểm 21/10, không phải lần kiểm lại
> installation/account/runtime hay đọc lại toàn bộ paper ngày 22/10. Phần
> “trước thay đổi” không mô tả dirty state hiện tại. Không có cài skill mới
> nào được xác nhận bởi banner này.
>
> Nguồn hiện tại: [00_START_HERE](../00_START_HERE.md),
> [PROJECT_BRIEF](../PROJECT_BRIEF.md), [report.md](../report.md),
> [protocol](../06_REVIEW_PROTOCOL.md),
> [G3 domain contract](../07_G3_DOMAIN_CONTRACT.md),
> [sync record 22/10](../DOCS_SYNC_2026-10-22.md).
> Các mô tả historical/legacy của root report trong tài liệu cùng thời điểm
> phải đọc theo snapshot đó; `report.md` hiện tại đã được cập nhật làm hồ sơ
> tổng hợp, bản cũ được giữ trong Git history.

Phạm vi user xác nhận: tổng quan tài liệu cho đề tài hiện tại. Kết quả của lượt
này là protocol, search log và note nguồn đầu tiên; chưa phải toàn văn review.

## 1. Snapshot repo trước thay đổi

- Branch `main`, HEAD `35d9bd446dc997df6e73cc8956a2d4a1e1c2bfdd`.
- Các thay đổi có sẵn: `.obsidian/`, `04_THEORY.md`, `GPT-sources/` đều untracked;
  không sửa những mục này.
- Có 23 mục P01–P23 trong `05_READING_PLAN.md`, 7 entry BibTeX, 2 PDF local
  (LipCache, STBP), chưa có note `notes/P*.md` trước lượt này.
- `GATES.md` ghi G0/G1/G2 đã qua; đây là trạng thái do repo ghi nhận, không phải
  kết quả chạy lại training/data audit trong lượt này.

## 2. Skill nào thực sự đang có và đã dùng?

| Thành phần | Bằng chứng live | Trạng thái trong lượt này |
|---|---|---|
| TMB `orchestrator` | `/home/pro/.codex/skills/orchestrator/SKILL.md` trỏ vào package local | Đã đọc và dùng để định tuyến; không phải methodology khoa học |
| TMB local package | `/home/pro/.codex/skill-packages/trustmebro-local/INSTALLATION.json` | Có 5 skills: dedup-merge, layout-init, orchestrator, roadmap-planner, roadmap-sync |
| TMB hooks | `hooks_enabled: false` trong installation record | Không có bằng chứng hooks TMB đang chạy |
| TMB import provenance | Source ghi `/home/pro/Downloads/hackathon/src`, imported 09/10/2026 | Source cũ hiện không tồn tại; snapshot vẫn đọc được |
| TMB source local hiện có | `/home/pro/Downloads/intern/TrustMeBro/README.md` + root `skills/` | Game Pack; không có skill specific/literature-review/scientific-writing |
| `find-skills` | `/home/pro/.codex/skills/find-skills/SKILL.md` | Đã dùng leaderboard + 3 CLI search bên dưới |
| TMB `roadmap-planner` | Đã đọc SKILL.md để đánh giá khả năng áp dụng | Không chạy stage tạo PLAN/ROADMAP; task này dùng reading plan có sẵn |
| Scientific review skill | Không thấy trong catalog skill đang nạp hoặc các root skill local đã kiểm | Phần bổ sung còn thiếu; chưa cài hay tạo skill mới |

Canonical root của TMB source hiện có là `skills/`, `templates/`, `rules/`,
`scripts/`, `hooks/`; `src/` là compatibility mirror theo README live. Không
đồng nhất source Game Pack hiện tại với snapshot generic đã import. Protocol
của research là file trong project, không phải skill TMB mới được cài.

## 3. Stack nghiên cứu đang dùng

Kiểm metadata ngay trong `.venv`, không suy từ lock file:

| Thành phần | Hiện trạng |
|---|---|
| Python | 3.11.15, executable `.venv/bin/python` |
| ML | torch 2.14.1+cpu; backbone 1D CNN, K=6 theo config/source |
| Data/plots | numpy 2.4.6, pandas 3.0.6, scipy 1.17.1, matplotlib 3.11.2 |
| Literature | Markdown, `refs.bib`, local PDF, note template; có config Obsidian và LaTeX source |
| auto_LiRPA | Được đề xuất trong docs; distribution `auto-lirpa` chưa cài trong project venv |
| bibtexparser | Chưa cài trong project venv |
| Literature search ở lượt này | Web search + primary pages/PDF; không có direct database export |

Không thấy integration Zotero/Elicit/scite/MLflow/W&B trong các file workflow,
requirements và scripts đã kiểm. Đây không phải audit mọi app/account của user;
không suy ra các dịch vụ đó không được dùng ở nơi khác. Có file `.tex` không
chứng minh compiler đang hoạt động hay đã có PDF hoàn chỉnh.

## 4. Search skill hiện tại

Các lệnh đã chạy, đều exit 0:

```text
npx --yes skills find literature-review
npx --yes skills find scientific-writing
npx --yes skills find systematic-literature-review
```

Ứng viên sát task nhất đã mở source chính chủ:

| Skill | Vai trò đánh giá | Snapshot discovery |
|---|---|---|
| K-Dense `literature-review` | Search/screen/extract/synthesize/citation workflow | CLI hiển thị 2.3K installs |
| K-Dense `scientific-writing` | Evidence-to-prose, consistency và metadata discipline | CLI hiển thị 2.4K installs |

Source: [literature-review SKILL.md](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/literature-review/SKILL.md),
[scientific-writing SKILL.md](https://github.com/K-Dense-AI/scientific-agent-skills/blob/593377bb048752abb0152deab9fbb1083a6087df/skills/scientific-writing/SKILL.md).
GitHub API tại lúc kiểm: repo có 53,416 stars, không archived;
`main` SHA `593377bb048752abb0152deab9fbb1083a6087df`.

Số installs/stars chỉ là tín hiệu discovery thay đổi theo thời điểm, không chứng
minh phương pháp đúng. Không dùng leaderboard và CLI như hai snapshot thống kê
có cùng thời điểm. Các candidate khác được search trả về nhưng chưa đọc source
đủ để khuyến nghị.

Đánh giá áp dụng: K-Dense là nguồn tham khảo workflow, **chưa cài/nạp/chạy toàn
bộ skill**. `literature-review` còn tham chiếu `parallel-cli` và yêu cầu figure
AI, chưa phù hợp để mặc định áp toàn bộ vào task này. Giữ công cụ search đang
có và protocol evidence trong repo; hình vẽ chỉ tạo khi có nội dung cần diễn đạt.

## 5. Kết quả pilot và việc chưa xong

Đã mở [NoScope proceedings PDF](https://www.vldb.org/pvldb/vol10/p1586-kang.pdf)
và [metadata Stanford](https://dawn.cs.stanford.edu/publications/noscope).
Xem [P04 note](P04_NoScope.md) để biết chính xác claim/locator được kiểm; chưa
reproduce code hay nhập số speedup.

Sửa metadata P03 theo [arXiv submission history](https://arxiv.org/abs/2106.12732):
v1 ngày 24/06/2021, latest-listed v2 ngày 03/02/2023. Không chuyển metadata check
thành trạng thái đã đọc sâu paper P03.

Các điểm cần tiếp tục kiểm được phát hiện từ primary sources trong pilot:

- P08 CROWN: đọc footnote 2, §3.1, PDF p.5 trước khi viết class-pair contribution:
  [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2018/file/d04863f100d59b3eb688a11f95b0ae60-Paper.pdf).
- P10 Fast-Lin/Fast-Lip: tên/thứ tự tác giả khác giữa
  [PMLR metadata](https://proceedings.mlr.press/v80/weng18a.html) và
  [PDF](https://proceedings.mlr.press/v80/weng18a/weng18a.pdf); cần ghi provenance
  khi thêm bibliography, không ghép hai thứ tự.
- P01 LipCache: cần giữ version/format locator rõ khi so HTML Roman numbering
  và PDF decimal numbering. Chưa tìm kiếm đầy đủ để kết luận mới nhất/duy nhất.

Chưa có screening toàn corpus, full-text extraction cho P01–P23, ma trận chất
lượng hoàn chỉnh, systematic coverage hoặc bài review cuối. C11/C12 vẫn mở.

## 6. Dẫn đường

- [Protocol](../06_REVIEW_PROTOCOL.md)
- [Search log có exact queries và giới hạn](2026-10-21_SEARCH_LOG.json)
- [Reading plan P01–P23](../05_READING_PLAN.md)
- [Claim ledger](../02_LITERATURE.md)

CREATED/VERIFIED chỉ áp dụng với file và check được nêu. Chưa có user acceptance.
