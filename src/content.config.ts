import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// fieldkit module reference markdown lives outside src/ in the package
// directory so the Python repo and the docs site stay in sync. Order
// matters for nav rendering — capabilities first, then nim, rag, eval,
// training, lineage, quant, publish, cli. `quant` + `publish` were
// added in v0.4 for the G3 GGUF publisher pick (MTBM Pick #1) and
// underpin the Phase 2 sync contract — `publish` writes per-artifact
// manifests into `src/content/artifacts/<slug>.yaml`.
export const FIELDKIT_MODULES = ['capabilities', 'nim', 'rag', 'eval', 'training', 'lineage', 'quant', 'publish', 'cli'] as const;

// Articles live at ../articles/<slug>/article.md and are authored via the
// tech-writer skill. We keep that authoring workflow by loading articles
// from outside src/ with a glob loader, and collapse the id to the folder
// slug so URLs are /articles/<slug>/ rather than /articles/<slug>/article/.

export const STAGES = [
  'foundations',
  'training',
  'fine-tuning',
  'inference',
  'deployment',
  'agentic',
  'observability',
  'dev-tools',
] as const;

// Editorial series — the running narrative threads. An article belongs to
// at most one series. Preamble pieces outside the arc system leave it
// unset. Foundations covers F1–F7 and the bridge (B). The three application
// arcs follow the bridge; Looking Beyond Spark is the fourth, opportunistic
// thread for arithmetic that extrapolates beyond the 128 GB Spark envelope.
// "Machine that Builds Machines" was renamed from "Autoresearch" on
// 2026-05-08 and broadened to cover the full /book/ Part-4 Vision thesis —
// AI systems that build, improve, evaluate, or supervise other AI/ML
// artifacts. The slug stays "autoresearch" so existing inbound links and
// /series/autoresearch/ bookmarks keep resolving.
export const SERIES = [
  'Foundations',
  'Second Brain',
  'LLM Wiki',
  'Machine that Builds Machines',
  'Looking Beyond Spark',
  'Frontier Scout',
] as const;

// Slug-safe form for series, used in /series/<slug>/ URLs and the filter
// component. Mirror order with SERIES so chip rendering matches.
export const SERIES_SLUGS: Record<(typeof SERIES)[number], string> = {
  'Foundations': 'foundations',
  'Second Brain': 'second-brain',
  'LLM Wiki': 'llm-wiki',
  'Machine that Builds Machines': 'machine-that-builds-machines',
  'Looking Beyond Spark': 'looking-beyond-spark',
  'Frontier Scout': 'frontier-scout',
};

export const SERIES_BY_SLUG: Record<string, (typeof SERIES)[number]> =
  Object.fromEntries(
    Object.entries(SERIES_SLUGS).map(([name, slug]) => [slug, name as (typeof SERIES)[number]]),
  );

const articles = defineCollection({
  loader: glob({
    pattern: '*/article.md',
    base: './articles',
    generateId: ({ entry }) => entry.split('/')[0],
  }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    author: z.string().default('Manav Sehgal'),
    product: z.string(),
    stage: z.enum(STAGES),
    difficulty: z.enum(['beginner', 'intermediate', 'advanced']),
    time_required: z.string(),
    hardware: z.string().default('NVIDIA DGX Spark'),
    tags: z.array(z.string()),
    summary: z.string().max(300),
    // Name (not path) of a signature figure component under
    // src/components/svg/. Rendered as the card's thumbnail on the home
    // and stage-filter pages. Optional — cards without one show no aside.
    signature: z.string().optional(),
    // Lifecycle. `published` = written and live. `upcoming` = placeholder
    // preview with an abstract; rendered dimmed with an "Upcoming" badge
    // and excluded from the home index (it still appears on its stage page
    // so readers see what is coming next).
    status: z.enum(['published', 'upcoming']).default('published'),
    // Articles frequently span more than one stage (e.g. a foundations
    // piece that also installs dev-tools). `stage` stays the primary
    // bucket; `also_stages` lists secondary buckets so the article shows
    // up on those stage pages too.
    also_stages: z.array(z.enum(STAGES)).default([]),
    // Editorial series — the running narrative thread the article belongs
    // to. Optional: preamble pieces outside the arc system leave it unset.
    series: z.enum(SERIES).optional(),
    // Which `/book/` chapter(s) this article grounds with field evidence.
    // Optional and mostly used by "Machine that Builds Machines" articles
    // (default [10]). The destination site can render a "Field evidence"
    // backlink at the foot of /book/<chapter>/ pages by querying articles
    // whose book_chapters includes the chapter number. Source repo doesn't
    // render this — the field is forward-compatible declaration.
    book_chapters: z.array(z.number().int().min(1).max(14)).optional(),
    // Which `fieldkit` modules this article exercises. Drives the
    // "uses fieldkit.X" chip on cards and the back-links from module
    // reference pages. Conservative — only set on articles that actually
    // import the module, not articles that merely mention it.
    fieldkit_modules: z.array(z.enum(FIELDKIT_MODULES)).optional(),
  }),
});

const fieldkit_docs = defineCollection({
  loader: glob({
    pattern: '*.md',
    base: './fieldkit/docs/api',
    generateId: ({ entry }) => entry.replace(/\.md$/, ''),
  }),
  schema: z.object({
    module: z.enum(FIELDKIT_MODULES),
    title: z.string(),
    summary: z.string(),
    order: z.number().int(),
  }),
});

// Phase 2 of the sync contract — per-artifact YAML manifests. `fieldkit.publish`
// writes one of these per HF push; the source repo validates the schema, the
// Mac destination renders `/artifacts/<kind>/` catalog pages from
// `getCollection('artifacts')`. Schema mirrors `ArtifactManifest.to_dict()`
// in `fieldkit/src/fieldkit/publish/__init__.py`.
export const ARTIFACT_KINDS = [
  'quant',
  'lora',
  'adapter',
  'embed',
  'reranker',
  'dataset',
  'space',
  'bench',
] as const;

const artifacts = defineCollection({
  loader: glob({
    pattern: '*.yaml',
    base: './src/content/artifacts',
    generateId: ({ entry }) => entry.replace(/\.yaml$/, ''),
  }),
  schema: z.object({
    slug: z.string(),
    kind: z.enum(ARTIFACT_KINDS),
    class: z.string(),
    base_model: z.string(),
    hf_repo: z.string(),
    variants: z.array(z.string()).default([]),
    perplexity: z.record(z.string(), z.number()).optional(),
    spark_tokens_per_sec: z.record(z.string(), z.number()).optional(),
    sustained_load_minutes: z.number().optional(),
    vertical_eval: z.record(z.string(), z.number()).optional(),
    vertical_eval_name: z.string().optional(),
    lineage_run_id: z.string().optional(),
    license: z.object({
      tier: z.string().default('free'),
      commercial_tier: z.string().optional(),
      // Upstream model license tag — `apache-2.0`, `llama2`, `cc-by-nc-4.0`,
      // etc. Mirrors the HF README frontmatter `license:` scalar so the
      // catalog page and HF badge stay in sync.
      model: z.string().optional(),
    }),
    article: z.string().optional(),
    civitai_id: z.number().int().optional(),
    download_count: z.number().int().optional(),
    published_at: z.string().optional(),
  }),
});

export const collections = { articles, fieldkit_docs, artifacts };
