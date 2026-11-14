"""Step 2b - non-US patents (EP, WO, CN, JP, KR, IN ...) from Google Patents Public Data.

One SQL query over `patents-public-data.patents.publications` picks every
publication with your areas' CPC codes, keeps one document per patent family
(granted first, then newest), caps each area at its target, and stores the result
in a staging table in YOUR project. The rows are then streamed to
corpus/patents/area-<name>/bigquery/part-NNNNN.jsonl (one patent per line).

BigQuery bills by bytes scanned (full-text columns are large). ALWAYS dry-run first.

    gcloud auth application-default login        # once
    python 03_download_bigquery.py               # dry run: prints the bytes it would scan
    python 03_download_bigquery.py --run         # run the query, then export
    python 03_download_bigquery.py --export-only PROJECT.DATASET.TABLE   # re-export
    python 03_download_bigquery.py --show-sql    # print the SQL
"""
import argparse
import datetime as dt
import json
import sys

from common import load_config, norm_cpc

SRC = "`patents-public-data.patents.publications`"


def area_case(areas):
    whens = []
    for a in areas:
        conds = " OR ".join(f"STARTS_WITH(REPLACE(c.code, ' ', ''), '{norm_cpc(p)}')" for p in a["cpc"])
        whens.append(f"WHEN EXISTS(SELECT 1 FROM UNNEST(cpc) c WHERE {conds}) THEN '{a['name']}'")
    return "CASE " + " ".join(whens) + " END"


def localized(field, english_only):
    en = f"(SELECT x.text FROM UNNEST({field}) x WHERE x.language = 'en' LIMIT 1)"
    if english_only:
        return en
    return f"COALESCE({en}, (SELECT x.text FROM UNNEST({field}) x LIMIT 1))"


def build_sql(cfg):
    b = cfg["bigquery"]
    eo = b.get("english_only", True)
    targets = " ".join(f"WHEN '{a['name']}' THEN {int(a.get('target', 1000000))}" for a in cfg["areas"])
    skip_us = ""
    if b.get("skip_families_with_us", True):
        skip_us = (f"AND family_id NOT IN (SELECT DISTINCT family_id FROM {SRC} "
                   f"WHERE country_code = 'US' AND family_id IS NOT NULL)")
    return f"""
WITH base AS (
  SELECT
    publication_number, country_code, kind_code, family_id,
    publication_date, priority_date, filing_date,
    {localized('title_localized', eo)}       AS title,
    {localized('abstract_localized', eo)}    AS abstract,
    {localized('claims_localized', eo)}      AS claims,
    {localized('description_localized', eo)} AS description,
    ARRAY(SELECT c.code FROM UNNEST(cpc) c)                 AS cpc,
    ARRAY(SELECT a.name FROM UNNEST(assignee_harmonized) a) AS assignees,
    {area_case(cfg['areas'])} AS area
  FROM {SRC}
  WHERE country_code IN UNNEST(@countries)
    AND publication_date >= @min_date
    {skip_us}
),
one_per_family AS (
  SELECT * FROM base
  WHERE area IS NOT NULL AND claims IS NOT NULL
  QUALIFY ROW_NUMBER() OVER (
    PARTITION BY COALESCE(family_id, publication_number)
    ORDER BY REGEXP_CONTAINS(kind_code, r'^[BC]') DESC,
             abstract IS NOT NULL DESC, publication_date DESC) = 1
)
SELECT * FROM one_per_family
WHERE TRUE   -- BigQuery needs a WHERE before QUALIFY
QUALIFY ROW_NUMBER() OVER (PARTITION BY area ORDER BY publication_date DESC)
        <= CASE area {targets} END
"""


def export(client, table_id, cfg, rows_per_file=2000):
    from collections import defaultdict
    corpus = cfg["paths"]["corpus"] / "patents"
    files, counts, buffers = defaultdict(int), defaultdict(int), defaultdict(list)

    def flush(area):
        if not buffers[area]:
            return
        files[area] += 1
        out = corpus / f"area-{area}" / "bigquery" / f"part-{files[area]:05d}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for rec in buffers[area]:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        buffers[area] = []

    table = client.get_table(table_id)
    print(f"Exporting {table.num_rows} rows from {table_id} ...")
    for i, r in enumerate(client.list_rows(table, page_size=1000), 1):
        rec = dict(r.items())
        area = rec.get("area") or "unassigned"
        buffers[area].append(rec)
        counts[area] += 1
        if len(buffers[area]) >= rows_per_file:
            flush(area)
        if i % 10000 == 0:
            print(f"  {i} rows", dict(counts))
    for area in list(buffers):
        flush(area)
    print("Done:", dict(counts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="actually run (default is a dry run)")
    ap.add_argument("--export-only", metavar="TABLE")
    ap.add_argument("--show-sql", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    b = cfg["bigquery"]
    sql = build_sql(cfg)
    if a.show_sql:
        print(sql)
        return

    from google.cloud import bigquery
    client = bigquery.Client(project=b["project"])

    if a.export_only:
        export(client, a.export_only, cfg)
        return

    params = [bigquery.ArrayQueryParameter("countries", "STRING", b.get("countries", ["EP", "WO"])),
              bigquery.ScalarQueryParameter("min_date", "INT64", int(b.get("min_pub_date", 20150101)))]

    dry = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params, dry_run=True, use_query_cache=False))
    tb = dry.total_bytes_processed / 1e12
    print(f"Dry run: this query will scan {dry.total_bytes_processed / 1e9:,.0f} GB ({tb:.2f} TB).")
    print("Check current on-demand pricing at cloud.google.com/bigquery/pricing before running.")
    if not a.run:
        print("Nothing was billed. Re-run with --run to execute.")
        return
    cap = int(b.get("max_bytes_billed", 3_000_000_000_000))
    if dry.total_bytes_processed > cap:
        sys.exit(f"Scan exceeds max_bytes_billed ({cap / 1e12:.2f} TB). Raise it in config.toml if intended.")

    ds = bigquery.Dataset(f"{b['project']}.{b['dataset']}")
    ds.location = "US"                       # the public dataset lives in the US
    client.create_dataset(ds, exists_ok=True)
    table_id = f"{b['project']}.{b['dataset']}.export_{dt.datetime.now():%Y%m%d_%H%M}"
    job = client.query(sql, job_config=bigquery.QueryJobConfig(
        query_parameters=params, destination=table_id,
        write_disposition="WRITE_TRUNCATE", maximum_bytes_billed=cap))
    print(f"Query running (job {job.job_id}) -> {table_id}. This can take 10-30 minutes ...")
    job.result()
    print(f"Billed {job.total_bytes_billed / 1e9:,.0f} GB.")
    export(client, table_id, cfg)
    print(f"\nStaging table kept at {table_id}; re-export with --export-only {table_id}")


if __name__ == "__main__":
    main()
