# Audits de traçabilité Paper 1 — compléments (2026-09-13)

Suite de l'audit principal. Les trois campagnes restantes sont traçables.
Aucune ne dépend des runs relancés (elles portent sur la phase d'identification
PINN, pas sur les runs LBM stationnaires).

## 08_temporal — TRACABLE
- Source : `results_reviewers/08_temporal/temporal_results.csv` + `mt1,mt3,mt5/{metadata.json,model.pt}`
- Runs : Re=500, Fourier, E*=0.25, seed 42, N_MT nominal 5, n_mt=1/3/5.
- Valeurs extraites : n_mt=1 -> Er=0.2390, f2=89.2%, ft=0.0% ;
  n_mt=3 -> Er=0.2778, f2=89.3%, ft=1.3% ;
  n_mt=5 -> Er=0.2533, f2=92.3%, ft=1.2%.
- Conformité : « temporal-mode enrichment Nt=1,3,5 at fixed Nx=6 leaves f2
  essentially unchanged and keeps the explicit time-dependent content below
  ~3% » (Responses_to_Editor_and_Reviewers.tex l.101). ft max 1.3% < 3%. OK.
- Ces runs alimentent la phrase SI : convergence temporelle = lock-in périodique
  + stabilisation des stats fenêtrées (supplementary.tex l.209), pas un résidu
  stationnaire -> AUCUN relance pour ce groupe (déjà acté dans pof1_relaunch_list.csv).

## 07_mode_count — TRACABLE
- Source : `results_reviewers/07_mode_count/mode_count_results.csv`
- Runs : mx4_mt3(12), mx6_mt1(6), mx6_mt3(18), mx6_mt5(30), mx8_mt5(40), mx8_mt9(72).
- Table tab:archmode (manuscript.tex l.1113-1130) :
  (4,3) 12  0.2586 91.2  = Er=0.258645 f2=0.91154  OK
  (6,1) 6   0.2390 89.2  = 0.239008  0.89188      OK
  (6,3) 18  0.2778 89.3  = 0.277757  0.89253      OK
  (6,5) 30  0.2533 92.3  = 0.253291  0.92336      OK
  (8,5) 40  0.2475 87.8  = 0.247462  0.87798      OK
  (8,9) 72  0.2775 87.3  = 0.277550  0.87294      OK
- Texte l.1097 : « f2 between 87.3% and 92.3% » = bornes réelles observées. OK.

## 02_loss_ablation — TRACABLE
- Source : `results_reviewers/02_loss_ablation/loss_ablation_results.csv` (14 configs + seed 42)
- Table tab:ablation (manuscript.tex l.1071-1080) :
  Baseline            0.2534 92.2 1.3  = 0.253394 0.921952 0.012539  OK
  No Lvar             0.2582 0.2  5.1  = 0.258191 0.002354 0.051036  OK
  No Lre              0.2534 92.2 1.3  = 0.253394 0.921952 0.012539  OK
  No Ldiss            0.2533 92.3 1.2  = 0.253300 0.923150 0.012428  OK
  No Lvar,Lre         0.2582 0.2  5.1  = 0.258191 0.002354 0.051036  OK
  No Lvar,Ldiss       0.2578 0.3  5.0  = 0.257795 0.002793 0.050108  OK
  lambda_var/2        0.2546 89.3 1.5  = 0.254572 0.892958 0.014919  OK
  2*lambda_var        0.2598 92.8 1.0  = 0.259829 0.927739 0.009833  OK
- Texte l.1043-1058 : branche 1er harmonique sous No-Lvar (f2~0.2%, ft~5.1%),
  f2=89.3-92.8% pour lambda_var *1/2 et *2. OK.
- « fourteen configurations » (Responses l.84) : 14 configs exactement dans le
  CSV. OK.

## Conclusion
Deux anomalies restantes pour ces trois campagnes : aucune. Blocage principal
inchangé : relance POF1 en cours.