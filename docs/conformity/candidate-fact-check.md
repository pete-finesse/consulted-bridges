# Candidate fact check (9 Nov 2026)

Source: `src/consulted_bridges/questions/proposed/candidates.py`, all 171 specs. For each I checked the first hop
(bridge), the second hop (answer), aliases or ambiguities that could change the answer, and whether the subject
carries the bridge or answer. I also checked the same-kind controls against the drafting rule "a word of the bridge's
kind with a DIFFERENT answer". The runner takes the wrong peer letter from `same_kind[0]`
(`conformity.wrong_letter`), so I also checked the rule "a wrong peer answer is not the answer of a plausible
alternative bridge" against that word. Facts I was less sure of I looked up on the web: Tiger Woods was born in
Cypress CA, Sampras in Washington DC, Karolinska's main campus is in Solna, Emory has been inside Atlanta since 1 Jan
2018, and Masada lies inside Israel's pre-1967 lines. All five are as drafted. `candidates.py` is not edited. The ids
stay as they are.

No bridge or answer is a wrong fact. Every problem below is in a control, an alias, or the wording.

Severity: **wrong** = a spec error that breaks a check (here, every one is a same-kind control with the bridge's own
answer); **ambiguous** = a plausible model answer or the run-time peer changes the reading; **cosmetic** = harmless
as it stands.

| id | subject | bridge | answer | problem | fix |
|---|---|---|---|---|---|
| C_LM_persepolis | the ruins of Persepolis | Iran | I | **wrong**: same-kind "Iraq" also starts with I. **Also ambiguous**: "Persia" (P) is the historic name of Iran, and the subject itself carries "Pers-" | drop |
| C_LM_babylon | the ruins of Babylon | Iraq | I | **wrong**: same-kind "Iran" also starts with I | same-kind ["Syria", "Jordan"] |
| C_LP_lennon | John Lennon | Liverpool | L | **wrong**: same-kind "Leeds" also starts with L | same-kind ["Manchester", "Birmingham"] |
| C_LP_graf | Steffi Graf | Mannheim | M | **wrong**: same-kind "Munich" also starts with M | same-kind ["Berlin", "Stuttgart"] |
| C_LU_ntu | National Taiwan University | Taipei | T | **wrong**: same-kind "Tokyo" also starts with T. **Also ambiguous**: the subject's "Taiwan" starts with T and shares "Tai-" with Taipei, so the model has a shortcut | drop |
| C_LU_cmu | Carnegie Mellon University | Pittsburgh | P | **wrong**: same-kind "Philadelphia" also starts with P | same-kind ["Cleveland", "Baltimore"] |
| C_LU_case | Case Western Reserve University | Cleveland | C | **wrong**: both same-kind words (Columbus, Cincinnati) start with C | same-kind ["Detroit", "Pittsburgh"] |
| C_LU_gonzaga | Gonzaga University | Spokane | S | **wrong**: same-kind "Seattle" also starts with S | same-kind ["Tacoma", "Portland"] |
| C_LU_marquette | Marquette University | Milwaukee | M | **wrong**: same-kind "Madison" also starts with M | same-kind ["Chicago", "Detroit"] |
| C_LP_jordan | Michael Jordan | Brooklyn | B | ambiguous: Brooklyn is a borough. The city is New York (N), and "New York" is the likely answer to "in which city" | drop |
| C_LP_marcopolo | Marco Polo | Venice | V | ambiguous: Korčula (K) birth claim, already flagged in `predict` | keep, flagged (drop if hop 1 is unsteady) |
| C_LM_suomenlinna | the Suomenlinna sea fortress | Finland | F | ambiguous: "Suomen-" means "Finland's" in Finnish, so the subject carries the bridge (same kind of leak as "Peking University"). The endonym Suomi starts with S | drop |
| C_LM_blarney | Blarney Castle | Ireland | I | ambiguous: "Republic of Ireland" (R) is a common name for the state | keep if hop 1 says "Ireland"; else drop |
| C_VC_blarney | Blarney Castle | Ireland | Yes | ambiguous: under "Republic of Ireland" the answer flips to No | same as above |
| C_LM_kilimanjaro | Mount Kilimanjaro | Tanzania | T | ambiguous: same_kind[0] "Kenya" is the classic wrong belief, so the wrong peer K is the answer of a plausible alternative bridge | reorder same-kind to ["Uganda", "Kenya"] |
| C_LM_pashupatinath | Pashupatinath Temple | Nepal | N | ambiguous: there is also a Pashupatinath Temple in Mandsaur, India, and same_kind[0] is "India", so the peer letter is I | reorder to ["China", "India"] |
| C_LP_kanye | Kanye West | Atlanta | A | ambiguous: raised in Chicago, and same_kind[0] "Chicago" makes the peer letter C | reorder to ["Memphis", "Chicago"] |
| C_LP_obama | Barack Obama | Honolulu | H | ambiguous: Chicago is his political home, and same_kind[0] "Chicago" makes the peer letter C | reorder to ["Seattle", "Chicago"] |
| C_LP_merkel | Angela Merkel | Hamburg | H | ambiguous: grew up in East Germany and her career was in Berlin. same_kind[0] "Berlin" makes the peer letter B | reorder to ["Munich", "Berlin"] |
| C_LP_hawking | Stephen Hawking | Oxford | O | ambiguous: strongly tied to Cambridge, and same_kind[0] "Cambridge" makes the peer letter C | reorder to ["London", "Cambridge"] |
| C_LP_haaland | Erling Haaland | Leeds | L | ambiguous: plays for Manchester City, and same_kind[0] "Manchester" makes the peer letter M | reorder to ["Oslo", "Manchester"] |
| C_LP_lebron | LeBron James | Akron | A | ambiguous: tied to Cleveland, and same_kind[0] "Cleveland" makes the peer letter C | reorder to ["Columbus", "Cleveland"] |
| C_LU_harvard | Harvard University | Cambridge | C | ambiguous: "Boston" is a common answer, and same_kind[0] "Boston" makes the peer letter B | reorder to ["Providence", "Boston"] |
| C_LU_georgetown | Georgetown University | Washington | W | ambiguous: Georgetown is the Washington neighbourhood the campus sits in (a separate town until 1871), so the model may answer "Georgetown" (G) | keep, flagged |
| C_SP_gretzky | Wayne Gretzky | hockey | 6 | ambiguous: "on the field ... in hockey" reads as field hockey (11 a side) outside North America | bridge "ice hockey" if one token, else drop |
| C_LU_karolinska | the Karolinska Institute | Stockholm | S | cosmetic: the main campus is in Solna, a separate municipality. Solna also starts with S, so the answer holds | keep |
| C_LU_bohr | the Niels Bohr Institute | Copenhagen | C | cosmetic: it is an institute of the University of Copenhagen, not a university, so "main campus" reads oddly | keep, or reword hop 1 to "located" |
| C_LU_lomonosov | Lomonosov State University | Moscow | M | cosmetic: same-kind "Kiev". Current English spelling is Kyiv (K either way) | "Kyiv" if one token, else keep |
| C_IN_perlman | Itzhak Perlman | violin | 4 | cosmetic: same-kind "piano" has about 230 strings, which is multi-digit and loosely defined | same-kind ["guitar", "mandolin"] |
| C_LM_chocolatehills | the Chocolate Hills | Philippines | P | cosmetic: usually said "the Philippines". A model that keeps the article could say T | keep, flagged |

**171 checked: 30 have an issue (9 wrong, 16 ambiguous, 5 cosmetic), 141 clean.** Each row is counted once, at its
worst severity. Recommended drops: persepolis, ntu, jordan, suomenlinna (4). The rest need a control swap or reorder,
or a watch flag. Every replacement same-kind word needs the one-token check before it goes in: same-kind words that
are not one token are skipped at run time.

Checked and judged clean, noted for the record: Turkey/Türkiye (T either way); Tour de France sometimes starts in the
last days of June (most of the race is always in July); Kentucky Derby moved to September once, in 2020; Indonesia
spans into Oceania, but Java is in Asia; Emory's address is Atlanta since the 2018 annexation; Tiger Woods was born in
Cypress, CA; Adele was born in Tottenham, which is in London; Schonbrunn and Belem are ASCII spellings.

## Outside the fact check (follow-ups, not fixed)

1. **Peer answers for the non-letter families.** `conformity.wrong_letter` returns a letter for the category
   families, because those answers are not in `COUNT_WRONG`. Examples: C_CT_petra (Asia) → peer says "E",
   C_DS_petra (Right) → "I", C_VC_petra (No) → "I", C_EO_fuji (Odd) → "F". `wrong_letter2` also returns a letter
   for the number families: C_MN_valentine → "M", C_EL_hydrogen → "H". The gate screen is solo, so it is not
   affected. The peer block on these families would be.
2. **Shared bridges across C_LP and C_LU.** Washington (sampras, georgetown) and Atlanta (kanye, emory) appear in
   both families with the same second hop, the first letter. `split()` groups by family before bridge, so these
   pairs can land in different halves, against the drafting rule.
