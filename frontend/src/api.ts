const API_BASE = import.meta.env.VITE_API_BASE_URL ?? '';

export type Customer = {
  customer_id: string;
  customer_name: string;
  customer_segment: string;
  customer_region: string;
  loyalty_tier: string;
  customer_age: number;
};

export type Recommendation = {
  product_id: string;
  product_name: string;
  category: string;
  brand: string;
  price: number;
  margin_pct: number;
  model_purchase_probability: number;
  item_similarity_score: number;
  final_ranking_score: number;
  expected_incremental_revenue: number;
  already_purchased: boolean;
  why: string[];
};

export type RecommendationPayload = {
  customer: Customer;
  model: {
    name: string;
    target: string;
    ranking_formula: string;
    features: Array<{ name: string; description: string }>;
  };
  recommendations: Recommendation[];
  portfolio_impact: {
    total_expected_incremental_revenue: number;
    average_model_probability: number;
    top_k: number;
  };
};

export type MetricsPayload = {
  selected_model: string;
  dataset: {
    rows: number;
    positive_rate: number;
    customers: number;
    products: number;
    target: string;
  };
  models: Record<string, Record<string, number>>;
  ranking_formula: string;
};

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function fetchCustomers() {
  return getJson<Customer[]>('/api/customers');
}

export function fetchRecommendations(customerId: string, topK = 8) {
  return getJson<RecommendationPayload>(`/api/recommendations/${customerId}?top_k=${topK}`);
}

export function fetchMetrics() {
  return getJson<MetricsPayload>('/api/ml/metrics');
}
