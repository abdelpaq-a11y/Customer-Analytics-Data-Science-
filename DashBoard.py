import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from dash_extensions.enrich import DashProxy, Output, Input, MultiplexerTransform

df = pd.read_csv(r"Customers_Fakedata_Cleaned.csv")
df.columns = df.columns.str.strip()
df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
df['Month'] = df['PurchaseDate'].dt.month

app = DashProxy(__name__, transforms=[MultiplexerTransform()], external_stylesheets=[dbc.themes.SOLAR])
app.layout = dbc.Container([
    html.H1("🌟 Customers Dashboard", className="text-center mt-4 mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Gender:"),
            dcc.Dropdown(
                options=[{'label': g, 'value': g} for g in sorted(df['Gender'].unique())],
                value=sorted(df['Gender'].unique()),
                multi=True,
                id='gender-filter'
            )
        ], width=4),
        dbc.Col([
            html.Label("Select Month:"),
            dcc.Dropdown(
                options=[{'label': m, 'value': m} for m in sorted(df['Month'].dropna().unique())],
                value=sorted(df['Month'].dropna().unique()),
                multi=True,
                id='month-filter'
            )
        ], width=4),
        dbc.Col([
            html.Label("Select Product Category:"),
            dcc.Dropdown(
                options=[{'label': c, 'value': c} for c in sorted(df['ProductCategory'].unique())],
                value=sorted(df['ProductCategory'].unique()),
                multi=True,
                id='category-filter'
            )
        ], width=4),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Sales"), html.H2(id='total-sales', className="animate-number")]), color="warning", inverse=True), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Average Purchase"), html.H2(id='avg-purchase', className="animate-number")]), color="danger", inverse=True), width=4),
        dbc.Col(dbc.Card(dbc.CardBody([html.H4("Total Records"), html.H2(id='total-records', className="animate-number")]), color="info", inverse=True), width=4),
    ], className="mb-4"),
    # Graphs with vibrant colors
    dbc.Row([
        dbc.Col(dcc.Graph(id='gender-bar'), width=6),
        dbc.Col(dcc.Graph(id='gender-pie'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='monthly-sales'), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='product-bar'), width=6),
        dbc.Col(dcc.Graph(id='age-hist'), width=6)
    ]),
    dbc.Row([
        dbc.Col(dcc.Graph(id='scatter-plot'), width=6),
        dbc.Col(dcc.Graph(id='heatmap'), width=6)
    ])
], fluid=True)
@app.callback(
    Output('total-sales', 'children'),
    Output('avg-purchase', 'children'),
    Output('total-records', 'children'),
    Output('gender-bar', 'figure'),
    Output('gender-pie', 'figure'),
    Output('monthly-sales', 'figure'),
    Output('product-bar', 'figure'),
    Output('age-hist', 'figure'),
    Output('scatter-plot', 'figure'),
    Output('heatmap', 'figure'),
    Input('gender-filter', 'value'),
    Input('month-filter', 'value'),
    Input('category-filter', 'value')
)
def update_dashboard(selected_genders, selected_months, selected_categories):
    # Filter dataframe
    dff = df[df['Gender'].isin(selected_genders) &
             df['Month'].isin(selected_months) &
             df['ProductCategory'].isin(selected_categories)]
    # KPI values with formatting
    total_sales = f"${dff['PurchaseAmount'].sum():,.0f}"
    avg_purchase = f"${dff['PurchaseAmount'].mean():.1f}" if not dff.empty else "$0"
    total_records = dff.shape[0]
    # Gender Bar
    fig_gender_bar = px.bar(dff, x='Gender', y='PurchaseAmount', color='Gender',
                            text_auto=True, title="Total Purchase by Gender", color_discrete_sequence=px.colors.qualitative.Vivid)
    # Gender Pie
    gender_counts = dff['Gender'].value_counts().reset_index()
    gender_counts.columns = ['Gender','Count']
    fig_gender_pie = px.pie(gender_counts, values='Count', names='Gender',
                            title="Gender Distribution", color_discrete_sequence=px.colors.sequential.Agsunset)
    # Monthly Sales Line
    monthly_sales = dff.groupby('Month')['PurchaseAmount'].sum().reset_index()
    fig_monthly = px.line(monthly_sales, x='Month', y='PurchaseAmount',
                          markers=True, title="Monthly Sales Trend", color_discrete_sequence=['#EF553B'])
    # Product Category Bar
    product_counts = dff['ProductCategory'].value_counts().reset_index()
    product_counts.columns = ['ProductCategory','Count']
    fig_product = px.bar(product_counts, x='ProductCategory', y='Count', color='ProductCategory',
                         title="Product Category Distribution", color_discrete_sequence=px.colors.qualitative.Set3)
    # Age Histogram
    fig_age = px.histogram(dff, x='Age', nbins=20, color='Gender', title="Age Distribution",
                           color_discrete_sequence=px.colors.qualitative.Pastel)
    # Scatter Plot
    fig_scatter = px.scatter(dff, x='Age', y='PurchaseAmount', color='Gender',
                             title="Age vs Purchase Amount", color_discrete_sequence=px.colors.qualitative.Bold)
    # Heatmap
    if not dff.empty:
        corr_matrix = dff[['Age','Rating','PurchaseAmount']].corr()
    else:
        corr_matrix = pd.DataFrame([[0,0,0],[0,0,0],[0,0,0]], columns=['Age','Rating','PurchaseAmount'], index=['Age','Rating','PurchaseAmount'])
    fig_heat = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap",
                         color_continuous_scale='RdBu_r')
    return total_sales, avg_purchase, total_records, fig_gender_bar, fig_gender_pie, fig_monthly, fig_product, fig_age, fig_scatter, fig_heat

if __name__ == "__main__":
    app.run(debug=True)

