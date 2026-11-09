# Brief for Codex: you are the writer

Paste this into Codex, in the repository folder `/Users/peter/code/ai-safety/consulted-bridges` (branch `conformity`).

---

You are the writer of a short research paper. Read `docs/paper/redteam-protocol.md` first and follow it exactly;
then `docs/paper/outline.md` and `docs/paper/resources.md`; then the sources they name. A Claude agent is drawing
the figures and will review your text each round; you review its figures each round. Peter is the editor.

Your job this round (round 1):
1. Write the title. It should say that we test *use* of the hidden step, not its presence, and name the group
   setting. The repository is called consulted-bridges; "consulted" is the project's word for a bridge the answer
   depends on.
2. Fill the Google Doc https://docs.google.com/document/d/1jG87fY_DJxMCtxkZ3t7jAh3VaMXYR49tqZlzh5sM918/edit with the
   full text of every section the outline marks SETTLED, in the outline's order, with a heading per section. For each
   PENDING RUN 3 item write one visible placeholder line: "[PENDING RUN 3: <what goes here>]". For each PETER item
   write your best version and mark the choice: "[PETER: <the alternative>]".
3. Where the outline names a figure, leave a line "[Figure N: <what it shows>; file docs/paper/figures/<name>]" so
   Claude can match it; Claude's `figures/README.md` lists the names.
4. Write `docs/paper/rounds/r1-codex.md`: the title and two alternatives; a table of every number you used with its
   source file and line; the PETER choices you made; questions for Peter. Commit only that file and the brief's
   outputs by name; never `git add -A`; never push.

Rules that override everything else: numbers and claims come only from the sources in the protocol; the wording
rules in the protocol's checklist apply to you as the writer; Peter's style is short sentences, plain words, every
term defined where it first appears, setup before result, no hype, no em-dashes. The earlier draft docx is for
framing only; its numbers are superseded.

When Peter says "Claude round 1 is in", read `docs/paper/rounds/r1-claude.md`, answer every finding in
`rounds/r2-codex.md` (accept with the fix, or reject with the reason and the source line), revise the Doc, and review
the figures in `docs/paper/figures/` into `rounds/r2-codex-figures.md` using the protocol's figure checklist.
