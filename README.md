# vWS: Virtual Win Shares for College Basketball

**[▶ Try the live app](https://datascience-7zcphr6hewugkdo4dda6hh.streamlit.app/)**

A player-value metric for NCAA Division I men's basketball and an interactive app for coaches, built in collaboration with John Andrzejek (then Head Coach at Campbell University, now Associate Head Coach at the University of Louisville).

## The Question

With NIL collectives and the transfer portal, programs need better answers to two questions:

1. **Compensation:** How much does a player's performance actually contribute to team success?
2. **Development:** Where does a roster need to improve to reach the next tier?

**Example:** Campbell entered the season ranked 203rd in Division I by [KenPom](https://kenpom.com/), with a goal of reaching 140th. The app compares the player value profile of a typical #203 team with a typical #140 team, showing how much value is needed at each roster spot. That helps with recruiting, transfer portal decisions, and player development.

## Results

| Metric | Correlation with team quality (KenPom AdjEM) |
|---|---|
| Traditional Win Shares | 0.776 |
| **vWS** | **0.875** |

Data: 15 seasons (2011–2025) of Division I player and team statistics. vWS also passes the eye test: its all-time leaders include Zach Edey, Doug McDermott, Kemba Walker, and Jimmer Fredette.

## Why Traditional Win Shares Falls Short

Win Shares (WS) is a familiar, intuitive metric for coaches. A team's players' Win Shares should add up to roughly its number of wins. Its key ingredient is Offensive Rating (ORtg), a player's points produced per 100 possessions.

The problem is that **low-usage players show far more variance in ORtg**, so both the most and least efficient players always appear at low usage (a "cone" shape when ORtg is plotted against usage; see `analysis/2_Develop_vWS.pdf`). That misses the premium college basketball places on high-usage players who stay efficient, the players who can create their own offense.

## Method

vWS rebuilds Win Shares with an adjusted offensive rating:

1. **Regress** ORtg on usage rate (USG%).
2. **Model the variance** by regressing squared residuals on USG%, which gives an estimated standard deviation at each usage level.
3. **Rescale** each player's ORtg: compute a z-score relative to the spread at *their* usage, then map it back using the spread at *average* usage (19%). High-usage players are stretched away from the regression line and low-usage players are shrunk toward it, producing roughly constant variance.
4. **Penalize** each percentage point of usage below average.
5. **Adjust for strength of schedule** using opponents' adjusted defensive efficiency.
6. **Recompute Win Shares** with the adjusted ORtg, plus a strength-of-schedule weight on Defensive Win Shares.

An empirical Bayesian shrinkage approach (similar to methods used for batting averages in baseball) was tried first. It over-shrank low-usage players, so the z-score rescaling above was used instead.

## The App

Enter a team's KenPom rank and the app shows the typical vWS of that team's best player, second-best player, and so on, using regression models of each roster spot's vWS against KenPom rank. Coaches can:

- Compare the value profiles of teams at any two ranks
- See how much the team would improve if a given player reached a particular value
- Compare how value is distributed across the top 10 players at different tiers

## Repository Structure

| Path | Contents |
|---|---|
| `FinalCBBapp.py` | Streamlit app |
| `short_wide_vWS_df_October.csv` | Team-level vWS table the app reads |
| `analysis/1_Description_and_Outline.pdf` | Project overview and outline |
| `analysis/KD_vWS.ipynb` · `analysis/2_Develop_vWS.pdf` | Part II: developing and validating vWS (math, plots, results) |
| `analysis/KD2_create_vWSDF.ipynb` | Part III: builds the team-level table used by the app |
| `analysis/cbb_kenpom_team_names_lookup_table.csv` | Maps school names between data sources |

## Data

Raw data is **not included** because of source licensing:

- **Player statistics:** [Sports Reference College Basketball](https://www.sports-reference.com/cbb/)
- **Team efficiency data:** [KenPom](https://kenpom.com/) (subscription required)

To rerun the analysis, place your own exports in `analysis/` as `cbb_full_data.csv` and `KenPomDataSince2004.csv`.

## Running It

**App:**

```bash
pip install -r requirements.txt
streamlit run FinalCBBapp.py
```

**Analysis** (also requires `pip install jupyter`), from the `analysis/` folder:

1. Run `KD_vWS.ipynb` (uncomment the final save cell) to produce player-level vWS.
2. Run `KD2_create_vWSDF.ipynb` (uncomment the final save cell) to produce `short_wide_vWS_df_October.csv`.

## A Note on Tools

The data work and the vWS metric (Parts I–III) were done with minimal AI assistance. The Streamlit app relied heavily on AI coding assistance, with the focus on prompting for visualizations that are clear and useful to a coaching staff.

## Author

Kieran Devlin ([github.com/kdev335](https://github.com/kdev335))
