# vWS Dashboard: College Basketball Player Value App

An interactive Streamlit app for exploring **virtual Win Shares (vWS)**, a player-value metric for NCAA Division I men's basketball built in collaboration with John Andrzejek (Associate Head Coach, University of Louisville).

vWS correlates more strongly with team quality than traditional Win Shares (0.875 vs. 0.776 across 2011–2025). The metric itself, and how it was built, is in the companion repository: **[kdev335/vWS](https://github.com/kdev335/vWS)**.

## What It Does

Pick a team's KenPom rank, and the app shows the typical value profile of a team at that level: how much vWS its best player, second-best player, and so on usually contribute. Coaches can use it to:

- **Compare tiers:** see what separates a typical #240 team from a typical #100 team, player by player.
- **Value a roster spot:** set a replacement-level vWS to see how much a given player adds over a replacement.
- **Plan recruiting and transfers:** find which roster positions need to improve to reach the next tier.

### Features

- **Individual Analysis:** distribution of vWS for a chosen roster position at a chosen team rank, fit with a cubic model of vWS against KenPom rank
- **Team Dashboard:** the full top-10 value profile for teams in each 20-rank band
- **Interactive Editor:** adjust vWS values for a hypothetical roster and compare it to typical teams
- Plotly or Matplotlib charts, adjustable histogram bins, and data downloads

## Running It

```bash
pip install -r requirements.txt
streamlit run FinalCBBapp.py
```

The app reads `short_wide_vWS_df_October.csv`, a team-level summary table produced by the notebooks in the [vWS repository](https://github.com/kdev335/vWS).

## Author

Kieran Devlin ([github.com/kdev335](https://github.com/kdev335))
