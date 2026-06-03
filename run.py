import gradio as gr
from inference import initiate_prediction

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

* {
    font-family: 'DM Sans', sans-serif !important;
}

.gradio-container {
    max-width: 960px !important;
    margin: auto !important;
}

footer {
    display: none !important;
}

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 32px 0 20px;
}

.app-title {
    font-size: 30px !important;
    font-weight: 600 !important;
    letter-spacing: -0.5px;
    color: #111827;
    margin-bottom: 6px !important;
}

.app-subtitle {
    font-size: 14px !important;
    color: #6b7280 !important;
    margin-bottom: 0 !important;
    line-height: 1.7;
}

/* ── Price highlight card ── */
.price-highlight {
    background: #f0f9ff;
    border: 1.5px solid #bae6fd;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 16px;
}

.price-icon-wrap {
    width: 48px;
    height: 48px;
    border-radius: 10px;
    background: #e0f2fe;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
}

.price-label {
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.7px;
    color: #0369a1 !important;
    margin-bottom: 4px !important;
}

.price-section label {
    font-size: 11px !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.7px !important;
    color: #0369a1 !important;
}

.price-section input[type="number"],
.price-section .number-output {
    font-size: 34px !important;
    font-weight: 600 !important;
    color: #0369a1 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 4px 0 0 !important;
    height: auto !important;
    letter-spacing: -0.5px;
}

.form-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 24px;
}

.form-card select,
.form-card input[type="range"] {
    border-radius: 8px !important;
}

.form-card label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #374151 !important;
}

.predict-btn {
    height: 50px !important;
    font-size: 15px !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    letter-spacing: 0.1px;
    background: #0284c7 !important;
    border-color: #0284c7 !important;
    color: #ffffff !important;
    margin-top: 8px;
}

.predict-btn:hover {
    background: #0369a1 !important;
    border-color: #0369a1 !important;
}

.slider-value {
    font-size: 13px;
    font-weight: 500;
    color: #374151;
    background: #f3f4f6;
    border-radius: 6px;
    padding: 2px 8px;
}
"""

with gr.Blocks(
    theme=gr.themes.Default(
        font=[gr.themes.GoogleFont("DM Sans"), "sans-serif"],
        radius_size="md",
        spacing_size="md",
    ),
    css=CUSTOM_CSS,
    title="Used Car Price Prediction",
) as app:

    # ── Header ──────────────────────────────────────────────
    gr.HTML("""
        <div class="app-header">
            <div class="app-title">🚗 Used Car Price Prediction</div>

            <div class="app-subtitle">
                Estimate the resale value of a vehicle using a Machine Learning model trained on real-world used car market data.
                <br><br>

                📩 For custom model development,
                AI automation, or enterprise AI solutions, get in touch:

                <br>

                <a href="mailto:badhanbarman@gmail.com">
                    badhanbarman@gmail.com
                </a>
            </div>
        </div>
    """)

    # ── Input form ───────────────────────────────────────────
    with gr.Group(elem_classes=["form-card"]):

        with gr.Row():
            with gr.Column():
                brand = gr.Dropdown(
                    choices=[
                        "Mahindra", "Skoda", "Maruti Suzuki",
                        "Hyundai", "MG", "Audi", "Toyota",
                        "Honda", "Tata", "Ford", "Chevrolet",
                        "BMW", "Volkswagen", "Jaguar",
                        "Renault", "Kia", "Range Rover", "Nissan",
                    ],
                    value="Maruti Suzuki",
                    label="Brand",
                )

                car_type = gr.Dropdown(
                    choices=["SUV", "Sedan", "Hatchback", "MPV", "Luxury"],
                    value="SUV",
                    label="Car Type",
                )

                transmission = gr.Dropdown(
                    choices=["Manual", "Automatic"],
                    value="Manual",
                    label="Transmission",
                )

                fuel_type = gr.Dropdown(
                    choices=["Petrol", "Diesel", "CNG", "Electric", "Hybrid"],
                    value="Petrol",
                    label="Fuel Type",
                )

            with gr.Column():
                owner = gr.Dropdown(
                    choices=["1st", "2nd", "3rd+"],
                    value="1st",
                    label="Ownership",
                )

                year = gr.Slider(
                    minimum=2000,
                    maximum=2026,
                    value=2020,
                    step=1,
                    label="Manufacturing Year",
                )

                kilometers = gr.Slider(
                    minimum=0,
                    maximum=300000,
                    value=30000,
                    step=1000,
                    label="Kilometers Driven",
                )

        predict_btn = gr.Button(
            "✦ Predict Price",
            variant="primary",
            elem_classes=["predict-btn"],
        )

    # ── Output ───────────────────────────────────────────────
    with gr.Group(elem_classes=["price-section"]):
        result = gr.Number(
            label="Estimated Resale Price (₹)",
            value=None,
            interactive=False,
            container=True,
        )

    predict_btn.click(
        fn=initiate_prediction,
        inputs=[
            brand,
            car_type,
            transmission,
            fuel_type,
            owner,
            year,
            kilometers,
        ],
        outputs=result,
    )

app.launch()