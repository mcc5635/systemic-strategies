# hJLP-gym: Systemic Crypto Strategy Research & Simulation Framework

hJLP-gym is a modular, research-driven framework for developing, simulating, and evaluating advanced trading strategies for crypto assets, with a focus on Bitcoin perpetual futures. It supports agent-based design, reinforcement learning, backtesting, and live-trading integration, and is built for extensibility and reproducibility.

---

## Repository Structure

```
hJLP-gym/
├── hummingbot/                # Core strategies, environments, and agents
│   └── strategy/
│       └── btc_perpetuals/    # BTC Perpetuals strategy module (see its README)
├── conf/                      # Example configuration files
├── dashboard/                 # Dashboard and visualization tools (Node/React)
├── logs/                      # Simulation and experiment logs
├── models/                    # Saved models, checkpoints, normalization files
├── BTC_Perpetuals.pdf         # Whitepaper/strategy documentation
├── equity_curve.png           # Example output/visualization
└── README.md                  # This file
```

---

## Core Modules

- **hummingbot/strategy/btc_perpetuals/**  
  Modular trend-following and RL strategies for BTC perpetuals.  
  *See [btc_perpetuals/README.md](hummingbot/strategy/btc_perpetuals/README.md) for details.*

- **conf/**  
  Example YAML configuration files for exchanges, strategies, and experiments.

- **dashboard/**  
  Web dashboard for monitoring, visualization, and analytics. Built with Node/React.

- **logs/**  
  Stores simulation logs, monitoring CSVs, and experiment outputs.

- **models/**  
  Contains trained model checkpoints, normalization files, and best/final models.

- **BTC_Perpetuals.pdf**  
  Whitepaper describing the BTC Perpetuals strategy, agent design, and research results.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/mcc5635/systemic-strategies.git
cd systemic-strategies
```

### 2. Set up the Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
*(If `requirements.txt` is missing, see module-specific README for dependencies.)*

### 3. Configure your experiment

Edit or copy a config file from `conf/` (e.g., `conf_coinbase_perpetual.yml`).

---

## Running Simulations

To run a BTC Perpetuals simulation:

```bash
python hummingbot/strategy/btc_perpetuals/train_btc_perpetuals.py --simulation --timesteps 10000
```

- Output logs and results will be saved in `logs/` and `models/`.
- For other strategies or modules, see their respective README files.

---

## Logging, Monitoring, and Results

- **logs/**: Contains simulation logs, monitoring CSVs, and evaluation outputs.
- **models/**: Stores model checkpoints, normalization files, and best/final models.
- **dashboard/**: Run the dashboard to visualize results and monitor experiments.

---

## Extending hJLP-gym

- Add new strategies or agents under `hummingbot/strategy/`.
- Add new configuration files to `conf/`.
- Extend the dashboard for new metrics or visualizations.

---

## References

- **BTC_Perpetuals.pdf**: Whitepaper and technical documentation for the BTC Perpetuals strategy.
- See module-level READMEs for agent and strategy details.

---

## License

MIT License (see LICENSE file).

---

## Acknowledgements

- Inspired by open-source trading and RL research communities.
- Built on top of Hummingbot, NumPy, pandas, and other open-source libraries.

---

**For detailed documentation on the BTC Perpetuals strategy, see:**  
[hummingbot/strategy/btc_perpetuals/README.md](hummingbot/strategy/btc_perpetuals/README.md) 