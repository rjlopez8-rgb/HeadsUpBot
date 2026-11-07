# Lab: testing and tuning our bot

Nothing in `lab/` is needed by the tournament. The bot that plays is `agent.py`,
plus `hero.py`, `pokerlib.py`, `fasteval.py` and `data/pt_table.npy`.

| File | What it does |
|---|---|
| `sim.py` | Plays duplicate matches in one process, with no HTTP. It gives the same result as `judge_engine.py` (checked on seed 42: identical bb/deal and interval). Thinking time is multiplied by `SIM_TIME_SCALE` (default 2) to pretend the tournament machine is slower. |
| `pool.py` | 15 opponents: simple styles, "AI-typical" Monte Carlo bots (including one that runs out of time bank), and weird ones (check-raiser, river bluffer, preflop shover, an adaptive exploiter). |
| `league.py` | Our bot against the pool, ranked by worst matchup first. |
| `tune.py` | Mutates our bot's Params and plays each mutant against the strong part of the pool **and against the current champion**. It promotes a mutant only if it beats the champion by more than the noise level. |

## Commands

```bash
pip install numpy fastapi uvicorn pytest

# our bot against the whole pool (about 50 s on 2 cores, faster with more cores)
python lab/league.py

# add a teammate's bot (any agent.py-style file) to the pool
python lab/league.py --extra ../teammate-repo/agent.py

# one tuning round (about 8 min on 2 cores; scales with core count)
python lab/tune.py --cands 10 --deals 600 --seeds 2
```

## Results so far
* hero v1 beat all 15 pool opponents (1,000 deals each).
* Official judge over HTTP, pinned to 1 core: 0 errors, about 2 s of thinking out of 8 per half.
* The first tuning round found no change bigger than noise. The champion against an
  identical copy of itself scores about ±0.1 bb/deal, so small knob changes can't be
  measured with 2 x 600 deals. Real gains need structural changes or many more seeds.
