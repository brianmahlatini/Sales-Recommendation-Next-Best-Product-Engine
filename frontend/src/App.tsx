import { useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  DollarSign,
  LineChart,
  PackageSearch,
  Sparkles
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import {
  Customer,
  MetricsPayload,
  RecommendationPayload,
  fetchCustomers,
  fetchMetrics,
  fetchRecommendations
} from './api';

type ModelRow = {
  name: string;
  roc_auc?: number;
  ndcg_at_5?: number;
  precision_at_5?: number;
};

const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
const money = (value: number) =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value);

export function App() {
  const [customers, setCustomers] = useState<Customer[]>([]);
  const [selectedCustomer, setSelectedCustomer] = useState('C001');
  const [recommendations, setRecommendations] = useState<RecommendationPayload | null>(null);
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([fetchCustomers(), fetchMetrics()])
      .then(([customerData, metricData]) => {
        setCustomers(customerData);
        setMetrics(metricData);
        if (customerData[0]) {
          setSelectedCustomer(customerData[0].customer_id);
        }
      })
      .catch((err: Error) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selectedCustomer) return;
    fetchRecommendations(selectedCustomer, 8)
      .then(setRecommendations)
      .catch((err: Error) => setError(err.message));
  }, [selectedCustomer]);

  const chartData = useMemo(
    () =>
      recommendations?.recommendations.map((item) => ({
        name: item.product_id,
        probability: Math.round(item.model_purchase_probability * 1000) / 10,
        impact: item.expected_incremental_revenue,
        category: item.category
      })) ?? [],
    [recommendations]
  );

  const modelRows: ModelRow[] = metrics
    ? Object.entries(metrics.models).map(([name, values]) => ({ name, ...values }))
    : [];

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Sparkles size={22} />
          </div>
          <div>
            <h1>Next-Best-Product Engine</h1>
            <p>Supervised ML ranking dashboard</p>
          </div>
        </div>

        <label className="field-label" htmlFor="customer">
          Customer
        </label>
        <select
          id="customer"
          value={selectedCustomer}
          onChange={(event) => setSelectedCustomer(event.target.value)}
        >
          {customers.slice(0, 60).map((customer) => (
            <option key={customer.customer_id} value={customer.customer_id}>
              {customer.customer_id} · {customer.customer_segment} · {customer.loyalty_tier}
            </option>
          ))}
        </select>

        {recommendations && (
          <section className="customer-panel">
            <p className="eyebrow">Active Customer</p>
            <h2>{recommendations.customer.customer_name}</h2>
            <dl>
              <div>
                <dt>Segment</dt>
                <dd>{recommendations.customer.customer_segment}</dd>
              </div>
              <div>
                <dt>Region</dt>
                <dd>{recommendations.customer.customer_region}</dd>
              </div>
              <div>
                <dt>Loyalty</dt>
                <dd>{recommendations.customer.loyalty_tier}</dd>
              </div>
            </dl>
          </section>
        )}
      </aside>

      <section className="content">
        <header className="topbar">
          <div>
            <p className="eyebrow">Recommendation Systems + Customer Analytics</p>
            <h2>Customer-specific product rankings with ML-predicted conversion probability.</h2>
          </div>
          <div className="status-pill">
            <CheckCircle2 size={18} />
            Runtime model loaded
          </div>
        </header>

        {error && <div className="error">{error}</div>}

        {recommendations && metrics && (
          <>
            <section className="kpi-grid">
              <div className="metric-card">
                <DollarSign size={20} />
                <span>Expected impact</span>
                <strong>{money(recommendations.portfolio_impact.total_expected_incremental_revenue)}</strong>
              </div>
              <div className="metric-card">
                <BrainCircuit size={20} />
                <span>Selected model</span>
                <strong>{metrics.selected_model.replace(' stronger_model', '')}</strong>
              </div>
              <div className="metric-card">
                <Activity size={20} />
                <span>Positive rate</span>
                <strong>{percent(metrics.dataset.positive_rate)}</strong>
              </div>
              <div className="metric-card">
                <PackageSearch size={20} />
                <span>Ranking formula</span>
                <strong>ML + similarity + value</strong>
              </div>
            </section>

            <section className="analytics-grid">
              <div className="panel wide">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Top recommendations</p>
                    <h3>Predicted conversion probability</h3>
                  </div>
                  <BarChart3 size={20} />
                </div>
                <div className="chart-wrap">
                  <ResponsiveContainer width="100%" height={260}>
                    <BarChart data={chartData}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} />
                      <XAxis dataKey="name" />
                      <YAxis unit="%" />
                      <Tooltip />
                      <Bar dataKey="probability" radius={[6, 6, 0, 0]}>
                        {chartData.map((entry, index) => (
                          <Cell key={entry.name} fill={index % 2 === 0 ? '#206a5d' : '#f2a65a'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <div className="panel">
                <div className="panel-heading">
                  <div>
                    <p className="eyebrow">Model performance</p>
                    <h3>Evaluation metrics</h3>
                  </div>
                  <LineChart size={20} />
                </div>
                <div className="model-table">
                  {modelRows.map((row) => (
                    <div key={row.name} className="model-row">
                      <strong>{row.name.replace('Classifier ', 'Classifier').replace(' stronger_model', '')}</strong>
                      <span>ROC AUC {Number(row.roc_auc ?? 0).toFixed(3)}</span>
                      <span>NDCG@5 {Number(row.ndcg_at_5 ?? 0).toFixed(3)}</span>
                      <span>Precision@5 {Number(row.precision_at_5 ?? 0).toFixed(3)}</span>
                    </div>
                  ))}
                </div>
              </div>
            </section>

            <section className="recommendation-list">
              {recommendations.recommendations.map((item, index) => (
                <article key={item.product_id} className="recommendation-card">
                  <div className="rank">{index + 1}</div>
                  <div className="product-main">
                    <div>
                      <p className="eyebrow">{item.category} · {item.brand}</p>
                      <h3>{item.product_name}</h3>
                    </div>
                    <div className="score-stack">
                      <span>ML probability</span>
                      <strong>{percent(item.model_purchase_probability)}</strong>
                    </div>
                  </div>
                  <div className="product-metrics">
                    <span>Final score <strong>{item.final_ranking_score.toFixed(3)}</strong></span>
                    <span>Similarity <strong>{item.item_similarity_score.toFixed(3)}</strong></span>
                    <span>Impact <strong>{money(item.expected_incremental_revenue)}</strong></span>
                    <span>Price <strong>{money(item.price)}</strong></span>
                  </div>
                  <div className="reason-list">
                    {item.why.map((reason) => (
                      <span key={reason}>{reason}</span>
                    ))}
                  </div>
                </article>
              ))}
            </section>

            <section className="panel">
              <div className="panel-heading">
                <div>
                  <p className="eyebrow">Supervised ML design</p>
                  <h3>Features and target</h3>
                </div>
                <BrainCircuit size={20} />
              </div>
              <p className="model-copy">{recommendations.model.target}</p>
              <div className="feature-grid">
                {recommendations.model.features.map((feature) => (
                  <div key={feature.name}>
                    <strong>{feature.name}</strong>
                    <span>{feature.description}</span>
                  </div>
                ))}
              </div>
            </section>
          </>
        )}
      </section>
    </main>
  );
}
