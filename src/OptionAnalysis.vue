<template>
    <div id="app">
        <div class="top">
            <div><router-link to="/">home</router-link></div>
        </div>
        <div class="option-analysis">
            <form class="form">
                <div class="form-group"></div>
            </form>
            <div class="viz line-chart"></div>
        </div>
    </div>
</template>

<script>
import _ from "lodash";
// import * as vega from "vega";
// import { default as vegaEmbed } from "vega-embed";

class Option {
    constructor(date, strike, ask, quantity) {
        this.date = date;
        this.strike = strike;
        this.ask = ask;
        this.quantity = quantity;
    }

    valueAtPrice(price) {
        return price * 0;
    }
}

export class CallOption extends Option {
    valueAtPrice(price) {
        const premium = this.ask * this.quantity;
        if (price > this.strike) {
            const traded = (price - this.strike) * this.quantity;
            return -premium + traded;
        }
        return -premium;
    }
}

export class Asset {
    constructor(symbol, price) {
        this.symbol = symbol;
        this.price = price;
    }
}

export class OptionsSimulator {
    constructor(asset) {
        this.asset = asset;
        this.rows = [];
    }

    add(option) {
        this.rows.push(option);
        return this;
    }

    inferPriceRange() {
        const strikes = this.rows.map((o) => o.strike);
        const average = _.mean(strikes);
        const width = average * 0.25;
        const min = _.min(strikes) - width;
        const max = _.max(strikes) + width;
        return [min, max];
    }

    run(priceRange) {
        const samples = 100;
        const data = [];

        for (let sample = 0; sample < samples; sample++) {
            const price = priceRange[0] + ((priceRange[1] - priceRange[0]) / samples) * sample;
            const value = _(this.rows)
                .map((o) => o.valueAtPrice(price))
                .sum();
            data.push({ x: price, y: value });
        }

        const range = [_.min(data.map((d) => d.y)) * 1.2, _.max(data.map((d) => d.y)) * 1.2];

        console.log("sim:range", range);
        console.log("sim:data", data);

        return { data, range };
    }
}

export default {
    name: "OptionAnalysis",
    data() {
        return {
            form: {},
        };
    },
    async mounted() {
        const simulation = new OptionsSimulator(new Asset("AAPL", 170)).add(new CallOption(new Date(), 165, 5.5, 900));

        console.log(`sim:raw`, simulation);

        const priceRange = simulation.inferPriceRange();

        const { data, range: plRange } = simulation.run(priceRange);

        console.log("sim:data", data);
        console.log("sim:data", plRange);

        /*
        const lineSpec = {
            $schema: "https://vega.github.io/schema/vega-lite/v5.json",
            data: { name: "table" },
            width: 400,
            encoding: {
                x: { field: "x", type: "quantitative", scale: { domain: priceRange } },
                y: { field: "y", type: "quantitative", scale: { domain: plRange } },
            },
            config: {
                axis: {
                    labelColor: "#ffffff",
                },
                background: "#111111",
            },
            mark: { type: "line", tooltip: true },
        };

        const embedded = await vegaEmbed(".line-chart", lineSpec, {
            renderer: "svg",
            tooltip: { offsetX: -50, offsetY: 50 },
            actions: { source: false, editor: false, compiled: false },
        });

        const changeSet = vega
            .changeset()
            .remove(() => true)
            .insert(data);

        embedded.view.change("table", changeSet).run();
        */
    },

    methods: {},
};
</script>

<style>
.option-analysis {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
}

.buttons {
    display: flex;
    justify-content: space-between;
}
</style>
