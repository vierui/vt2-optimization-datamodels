                  ┌─────────────────────────┐
                  │  Full-Year Data (8760h) │
                  │ (Load, Wind, Solar, etc.)  
                  └─────────────────────────┘
                              │
                              │  1) Extract 3 Representative Weeks 
                              ▼
      ┌───────────────────────┬─────────────────────────┬───────────────────────┐
      │Week 2 (Winter, 168h)  │  Week 31 (Summer, 168h) │  Week 43 (Spring/Aut.,168h)
      │   Weighted ×13        │     Weighted ×13        │       Weighted ×26
      └──────────┬────────────┴──────────┬──────────────┴───────────┬─────────┘
                 │                       │                           │
                 │                       │                           │
                 └───────────────────────┼───────────────────────────┘
                                         │
                                         │  2) Combine into 3 "Blocks" (504 hours total)
                                         ▼
                             ┌─────────────────────────┐
                             │  Seasonal Blocks Data   │
                             │  (3 blocks × 168h each) │
                             │  Weighted in Objective  │
                             └─────────────────────────┘
                                         │
                                         │  3) DCOPF / Investment Model
                                         ▼
                     ┌──────────────────────────────────────┐
                     │ DCOPF / Investment DCOPF Optimization│
                     │ - Storage reset between blocks       │
                     │ - Weighted cost function             │
                     │ - Single run for all 3 blocks        │
                     └──────────────────────────────────────┘
                                         │
                                         │  4) Results & Post-Processing
                                         ▼
                     ┌──────────────────────────────────────┐
                     │ Final Year-Equivalent Results        │
                     │ (Cost, Dispatch, Investment, etc.)   │
                     │ Weighted to represent 52 weeks total │
                     └──────────────────────────────────────┘