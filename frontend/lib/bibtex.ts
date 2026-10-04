/* BibTeX entries built from paper metadata. arXiv papers get eprint fields, so reference managers
   and LaTeX link them back to arXiv. */

const ARXIV_ID = /^\d{4}\.\d{4,5}(v\d+)?$/;

export interface BibSource {
  id: string;
  title: string;
  authors: string[];
  year: number | null;
  categories?: string[];
}

/** Google Scholar style key: first author's surname + year + first meaningful title word, e.g. vaswani2017attention. */
function citeKey(p: BibSource, taken: Set<string>) {
  const surname = (p.authors[0] ?? "anon").trim().split(/\s+/).at(-1) ?? "anon";
  const word = p.title
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .find((w) => w.length > 3 && !["with", "from", "that", "this", "towards"].includes(w)) ?? "paper";
  const base = `${surname.toLowerCase().normalize("NFKD").replace(/[^a-z]/g, "") || "anon"}${p.year ?? ""}${word}`;
  let key = base;
  for (let i = 2; taken.has(key); i++) key = `${base}${String.fromCharCode(96 + i)}`; // ...b, ...c on collisions
  taken.add(key);
  return key;
}

// Characters LaTeX treats specially; braces keep the title's capitalisation (e.g. "BERT")
const escape = (s: string) => s.replace(/([&%$#_])/g, "\\$1");

export function toBibtex(papers: BibSource[]): string {
  const taken = new Set<string>();
  return papers
    .map((p) => {
      const arxiv = ARXIV_ID.test(p.id);
      const fields: [string, string][] = [
        ["title", `{${escape(p.title)}}`],
        ["author", escape(p.authors.join(" and ") || "Unknown")],
      ];
      if (p.year) fields.push(["year", String(p.year)]);
      if (arxiv) {
        fields.push(["eprint", p.id], ["archivePrefix", "arXiv"]);
        if (p.categories?.[0]) fields.push(["primaryClass", p.categories[0]]);
        fields.push(["url", `https://arxiv.org/abs/${p.id}`]);
      }
      const body = fields.map(([k, v]) => `  ${k} = {${v}}`).join(",\n");
      return `@misc{${citeKey(p, taken)},\n${body}\n}`;
    })
    .join("\n\n");
}

/** Offer text as a file download in the browser. */
export function downloadText(filename: string, text: string, type = "application/x-bibtex") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  a.click();
  URL.revokeObjectURL(url);
}
