# E10 — Protocolo Confirmatório da Mediação Episódica de Alerta (EAML)

**Versão 3.0 — emenda pré-holdout (ver Seção 13). O holdout permanece
intocado e não foi inspecionado.**

**Versão 2.0 — emenda pré-holdout (ver Seção 12).**

**Versão 1.0 — pré-declarada antes de qualquer execução.**
Continuação da numeração experimental do projeto (E1–E8 auditoria; E9 EGA no
artigo UCC2026). Este documento é congelado junto com a especificação
(`spec.json` + manifesto SHA-256) antes do laço de holdout.

---

## 1. Pergunta e hipóteses

**Pergunta:** a mediação episódica completa (EAML: janela W, R_E composto,
persistência/escalada, evidência de supressão OSS_t e
force_compute_before_alert) melhora a comunicação de risco em relação (a) ao
disparo legado frame-level com cooldown+histerese e (b) à EGA mínima validada
em E9 — sem perder recall de eventos críticos e dentro de um orçamento de
alertas declarado?

Hipóteses pré-declaradas (cada uma pode ser refutada; conjunto admissível
vazio é desfecho legítimo):

- **H1 (mediação):** EAML reduz falsos alertas e latência efetiva de alerta
  (TTA_ef) vs. o legado, com recall não-inferior (margem 0,02).
- **H2 (episódico > evento mínimo):** EAML é não-inferior à EGA em recall e
  TTA_ef, e superior em pelo menos um de: falsos alertas, alertas redundantes,
  ou taxa de alerta com evidência requentada (alertas emitidos com
  reused_ratio > 0,60 na janela).
- **H3 (valor da OSSL — ablação):** remover OSS_t da janela (braço A3)
  degrada a qualidade da evidência: mais alertas com evidência requentada
  e/ou pior TTA_ef em episódios com alta supressão.
- **H0 explícita:** se nenhum braço candidato passar os portões, o legado é
  mantido e o resultado é reportado como negativo.

## 2. Braços experimentais

Todos os braços compartilham: o MESMO gerador de episódios, o MESMO modelo de
observação, o MESMO MHEG (τΔ, τH, max_skip da calibração vigente) e os MESMOS
sorteios exógenos (números aleatórios comuns; igualdade verificada por
asserção). A ÚNICA diferença é a camada de decisão de alerta.

| Braço | Decisão de alerta | Origem |
|---|---|---|
| **A0 · Legado** (referência fixa) | p_t ≥ θ_alert + cooldown=3 + histerese 0,05 | `alert_governance.py` (UCC as-is) |
| **A1 · EGA mínima** | FSM de evento sobre p_t corrente (θ_on/θ_off, realerta sob escalada/não-resposta, teto) | `event_guard.py` (E9) |
| **A2 · EAML completa** | janela W, R_E, RISCO∧GUARDA, OSS_t, force_compute_before_alert | `edge_runtime` |
| **A3 · EAML−OSSL** (ablação) | idem A2, mas frames suprimidos entram na janela SEM OSS_t (p carregado sem qualidade/decaimento; force_compute desativado) | ablação de A2 |

Proibição de vazamento: nenhum braço acessa informação futura; nenhum braço
conhece o rótulo do episódio; nomes de braços não aparecem em nenhuma equação
de desfecho.

## 3. Cenários comportamentais

Gerador: `synthetic_driver_risk_v7.py`/`teragen_v9` (T=96 frames = 10 s,
D=31–37 atributos), com as receitas completas:

- **Normais:** N01–N03 · **Atenção:** A01–A03 · **Alerta:** L01–L03
  (borderline — fonte primária de falsos alertas);
- **Críticos abruptos:** C01–C04 (onset súbito) · **Críticos progressivos:**
  C05–C10 (escalada lenta) — estratificação obrigatória no reporte
  (lição E5/Tabela II: médias entre regimes escondem regressões).
- **12 personas** P01–P12 (`sim/driver_personas.py`): `risk_mod`,
  responsividade e fadiga por persona.

Composição por seed: 12 personas × 20 viagens × 12 episódios = 2.880
episódios/seed, com mix de receitas por persona conforme
`cenarios_personas.json` (proporção fixa: 50% N, 15% A, 15% L, 10% C-abrupto,
10% C-progressivo).

## 4. Modelo de observação e resposta (comum a todos os braços)

- **Observação:** p_obs(t) = clip(y_frame(t) + ε_t · 0,10 · (1,2 −
  observabilidade da persona)), ε_t ~ N(0,1) pré-sorteado. (Sensibilidade
  opcional: repetir seeds 42–45 com as probabilidades reais do AF-KD de
  `pipeline_completo/results/metrics/frame_probs_*` — não entra nos portões.)
- **Entropia para o MHEG:** H_bin(p_obs) da última computação.
- **Resposta do motorista:** q = clip(responsividade − 0,4·fadiga_acum,
  0,05, 1) por episódio; sorteio uniforme pré-amostrado (idêntico entre
  braços). A resposta alimenta `observe_response` (A1/A2/A3).
- Sem mitigação de risco no desfecho (o alvo aqui é a qualidade da
  comunicação, não o outcome pós-ação — declarado como limitação).

## 5. Métricas

**Primárias (portões):**

1. **Recall de eventos** = fração de episódios críticos com ≥1 alerta
   entregue em [t_onset, T].
2. **TTA_ef** (s) = (t_alert − t_onset)·Δt + miss·T_max, T_max = 10 s
   (análogo do TTDef para alerta entregue).
3. **FAR** = alertas entregues em episódios N/A/L por viagem.
4. **Alertas/viagem** (orçamento B = 2,5 — mesmo valor de E9).

**Secundárias (reporte obrigatório, sem portão):** alertas redundantes/viagem;
%frames suprimidos; nº de force_compute_before_alert; **SAR** (stale-alert
rate: fração de alertas com reused_ratio > 0,60 na janela no instante da
entrega); recall estratificado abrupto/progressivo; TTA_ef estratificado.

## 6. Desenho, seeds e congelamento

- **Unidade de atribuição:** persona×seed (cluster); as diferenças pareadas
  são agregadas por cluster antes de qualquer inferência (nunca inferência em
  nível de episódio sem agrupamento — lição E1/E6).
- **Seeds de desenvolvimento:** {201, 202, 203, 204, 205} — calibração da
  grade e seleção de candidata.
- **Seeds de holdout:** {3001–3020} (20 seeds, 240 clusters persona×seed) —
  geradas uma única vez, jamais inspecionadas antes do congelamento.
- **Grade de candidatas da EAML (A2), congelada:**
  W ∈ {8, 12, 16} × m ∈ {2, 3} × θ_episode ∈ {0,65; 0,70; 0,75} ×
  guard = grade C1–C12 de E9 restrita a {C3, C8, C10} → 54 combinações.
  A1 usa a C8 exata promovida em E9 (sem re-otimização).
  A3 herda a configuração vencedora de A2 (ablação pura).
- **Fluxo:** (1) rodar dev seeds; (2) aplicar portões (Seção 7) por regime;
  (3) escrever `spec_e10.json` + SHA-256 de spec, código dos braços e
  datasets; (4) rodar holdout UMA vez; (5) reportar tudo, inclusive falhas.
  O holdout não retroalimenta a seleção. Congelamento programático
  (não é pré-registro externo — declarar como limitação).

## 7. Critérios de admissibilidade (portões pré-declarados)

Avaliados no holdout, por diferença pareada Δ = candidato − A0 (legado),
IC por cluster (t agrupado, 240 clusters); U90 = limite superior unilateral:

```
G1  U90(Δ TTA_ef)            < 0          (melhora latência efetiva)
G2  Δ Recall                 ≥ −0,02      (não-inferioridade)
G3  Alertas/viagem           ≤ 2,5        (orçamento absoluto)
G4  U90(Δ FAR)               ≤ 0          (não piora falsos alertas)
G4b [só A2 vs A1]  SAR(A2)   <  SAR(A1)   (valor específico da EAML)
```

Regras de decisão:

- **A2 promovida** sobre o legado sse G1∧G2∧G3∧G4 no agregado E em ambos os
  regimes (abrupto e progressivo) separadamente.
- **H2 confirmada** sse A2 passa G2 vs A1 (recall) e G4b.
- **H3 confirmada** sse A3 falha em pelo menos um portão que A2 passa, ou
  SAR(A3) > SAR(A2) com IC excluindo zero.
- **Conjunto vazio** em qualquer nível ⇒ manter o braço anterior (A0) e
  publicar o resultado negativo com a mesma proeminência.

## 8. Análise estatística

- Diferenças pareadas sob números aleatórios comuns (mesmas realizações
  exógenas; asserção de igualdade byte a byte das colunas exógenas entre
  braços — padrão E2/E8).
- IC 95% bilateral para reporte; U90 unilateral para portões; teste de
  Wilcoxon por seed como análise de robustez (padrão TERA).
- Estratificação obrigatória: abrupto vs progressivo; noite vs dia;
  reporte por persona no material suplementar (sem portão por persona —
  n insuficiente; declarar).
- Sem correção múltipla entre H1–H3 (hipóteses ordenadas hierarquicamente:
  H1 → H2 → H3; testa-se a seguinte só se a anterior passar).

## 9. Artefatos e reprodutibilidade

```
experiments/e10_eaml/
  scenarios.py      # ponte gerador v7/v9 + personas (X, y, onset por seed)
  arms.py           # A0/A1/A2/A3 sobre o MESMO stream de frames
  metrics.py        # recall, TTA_ef, FAR, SAR, custos
  run_dev.py        # grade + portões → spec_e10.json (congela)
  run_holdout.py    # 1 execução; saídas por braço×seed×persona
  outputs/          # CSVs, spec_e10.json, MANIFEST.sha256, figuras, latex
```

Saídas mínimas: `results_dev.csv`, `spec_e10.json`, `results_holdout.csv`,
`decisions_gates.csv`, tabela LaTeX principal (braços × métricas ± IC,
estratificada), figura trade-off (FAR × TTA_ef com fronteira), CDF de TTA.

## 10. Ameaças à validade (declaradas a priori)

1. Tudo é sintético; o modelo de observação é declarado, não aprendido —
   a sensibilidade com probabilidades AF-KD (seeds 42–45) mitiga mas não
   elimina.
2. Sem mitigação/outcome pós-ação: mede-se comunicação, não redução causal
   de risco (diferente de E9; não comparar magnitudes entre E9 e E10).
3. Orçamento B=2,5 e margens (0,02) herdados de E9 por consistência — são
   escolhas de engenharia, não constantes validadas com humanos.
4. Congelamento programático ≠ pré-registro externo.
5. Personas são perfis sintéticos; nenhuma alegação demográfica ou de
   equidade.
6. A EGA (A1) usa a C8 de E9 sem re-otimização — favorece A2 na comparação de
   calibração; mitigação: reportar também a melhor EGA da grade dev como
   análise secundária.

## 12. Emenda v2 (pré-holdout, 2026-08-05)

**Fato que motivou a emenda:** a execução de desenvolvimento da v1
(5 seeds × 12 personas × 20 viagens; 54 candidatas; resultados preservados em
`experiments/e10_eaml/outputs_v1_strict/`) produziu **conjunto admissível
vazio**: 54/54 candidatas passaram G3 (orçamento) e G4 (FAR) e 54/54 falharam
G1 (TTA_ef) e G2 (recall) contra o legado A0. Diagnóstico: o A0 opera SEM
orçamento (3,43 alertas/viagem, 37% acima de B) sobre críticos de cauda
persistente — recall 1,0 e TTA mínimo são estruturalmente inalcançáveis por
qualquer braço que respeite B=2,5. O objetivo v1 (bater o legado irrestrito em
latência sob orçamento) é inviável neste mix; os portões expuseram a
inviabilidade em vez de premiar um substituto — mesmo padrão do E9-estrito.

**Mudanças declaradas (antes de qualquer contato com as seeds de holdout):**

1. **Referência dos portões pareados (G1, G2, G4): A1 (EGA/C8)** — a linha de
   base *governada* validada em E9 — em vez do legado A0. Mede-se o valor
   incremental da mediação episódica sob o mesmo regime de orçamento
   (governado vs. governado). O A0 permanece nos reportes como referência
   descritiva do custo da liberdade irrestrita de alertas (recall 1,0 ao
   preço de 3,43 alertas/viagem e FAR 1,22).
2. **Grade de guarda recalibrada para o regime de episódios deste gerador**
   (12 episódios × 96 frames, 2 críticos/viagem, borderlines L cruzando
   θ_on): guardas construídas por θ_on ∈ {0,55; 0,60; 0,65} ×
   A_max ∈ {3, 4, 6} (off_gap 0,03, gap 2, escalada 0,10, persistência 1) —
   9 guardas × W{8,12,16} × m{2,3} × θ_episode{0,65; 0,70; 0,75} = 162
   candidatas. A1 permanece a C8 exata de E9, sem re-otimização.
3. **G3 inalterado**: orçamento absoluto B = 2,5 alertas/viagem.
4. **Hipóteses reformuladas**: H1' = a EAML é admissível vs. A1 (G1∧G2∧G3∧G4,
   agregado e por regime); H2' = G4b (SAR) mantido; H3 (ablação) inalterada.
   O contraste descritivo vs. A0 é reportado integralmente, inclusive onde o
   legado domina.

**Análise adicional pré-declarada (A-OSC):** o mix v1 não contém a patologia
que motiva a mediação (risco oscilante, cruzamentos transitórios, rajadas).
Após o holdout v2, uma análise secundária com receitas oscilantes
(`L04-osc`: risco flutuando ±0,12 em torno de θ_on; `C11-flutter`: crítico
com oscilação pré-onset) será executada com os MESMOS braços congelados —
sem re-seleção — para medir a sensibilidade das conclusões ao regime
temporal. Resultados reportados como análise adicional, não como portão.

## 13. Emenda v3 (pré-holdout, 2026-08-05) — fidelidade à H2 pré-declarada

**Fato que motivou:** o dev v2 (162 candidatas, `outputs_v2_superiority/`)
produziu conjunto vazio 162/162, mas o diagnóstico mostrou que o portão G1
implementado (superioridade estrita de TTA_ef vs. A1, U90 < 0) era **mais
estrito do que a hipótese H2 pré-declarada na Seção 1**, que sempre exigiu
apenas *não-inferioridade* em TTA_ef e recall vs. EGA, com *superioridade em
pelo menos uma* dimensão de qualidade (FAR, redundantes ou SAR). Os dados v2
mostram quase-paridade de latência (melhores candidatas: ΔTTA +0,21–0,42 s)
com 12 candidatas satisfazendo orçamento+recall+FAR.

**Mudança (v3):** os portões A2-vs-A1 passam a implementar a H2 literal:

```
G1n  U90(Δ TTA_ef)        < +1,0 s   (não-inferioridade; margem declarada)
G2   Δ Recall             ≥ −0,02    (não-inferioridade, por regime)
G3   Alertas/viagem       ≤ 2,5
G4   U90(Δ FAR)           ≤ 0
G5   superioridade em ≥1 de {SAR, redundantes/viagem, FAR}
     (IC95 excluindo zero)
```

**Declaração de transparência:** a margem de 1,0 s (10% de T_max; ordem de
uma janela W de mediação) foi fixada APÓS a inspeção dos dados de
desenvolvimento — o desenvolvimento é exploratório por definição neste
protocolo; a proteção contra sobreajuste é o holdout congelado, que permanece
não inspecionado. As três especificações (v1 estrita vs. legado; v2
superioridade vs. EGA; v3 = H2 fiel) serão reportadas juntas, com os
resultados de v1 e v2 preservados como resultados negativos.

## 11. Cronograma sugerido

| Passo | Duração estimada |
|---|---|
| `scenarios.py` + `arms.py` + `metrics.py` + testes de igualdade exógena | 1–2 dias |
| Execução dev (5 seeds × 54 candidatas) + portões + congelamento | ~meio dia |
| Execução holdout (20 seeds × 4 braços) | ~horas |
| Análise, tabelas, figuras, redação | 2–3 dias |
