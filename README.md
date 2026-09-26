# March Madness Predictive Analytics

An end-to-end machine learning analysis of NCAA March Madness tournament games using historical team efficiency metrics, matchup-level feature engineering, model evaluation, Vegas market comparisons, and 2026 bracket simulation.

This project is an independent reconstruction and extension of work originally developed for the 2026 Brandeis Datathon by Team ASA. The modeling pipeline, model configurations, evaluation results, betting analysis, and bracket simulations in this repository were rebuilt independently.

---

## Project Overview

The goal of this project was to answer three main questions:

1. How accurately can historical team metrics predict March Madness game outcomes?
2. How do machine learning models compare with traditional basketball rating systems?
3. When a model disagrees with the Vegas favorite, does that disagreement identify potentially valuable underdog opportunities?

The final pipeline analyzes tournament games from 2008–2025 and uses the historical data to generate matchup probabilities and full-bracket simulations for 2026.

---

## Dataset

The analysis combines multiple college basketball datasets containing:

- NCAA tournament matchups and results
- KenPom and Barttorvik efficiency metrics
- TeamRankings ratings
- Resume and strength metrics
- AP Poll information
- Vegas closing point spreads

After preprocessing, the project contains:

- 1,144 tournament matchup records
- 1,070 completed historical games through 2025
- 21 clean matchup-differential features
- 2026 team metrics for bracket simulation

Raw source datasets are not included in this repository.

---

## Feature Engineering

Each tournament matchup is converted into a team-vs-team feature vector.

For every metric:

```text
Matchup Feature = Team 1 Metric - Team 2 Metric