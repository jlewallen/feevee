<template>
    <div :class="classes">
        <div class="main">
            <div :class="'status ' + tagClasses"><div></div></div>
            <div class="header" v-bind:style="{ width: scaleChartW + 'px' }">
                <div class="symbol">
                    <a :href="stockChartsUrl" target="_blank">{{ symbol }}</a>
                </div>
                <span>
                    <a
                        v-bind:class="{
                            price: true,
                            percent: true,
                            negative: visible.negative,
                        }"
                        :href="seekingAlphaUrl"
                        target="_blank"
                        v-if="visible.percent_change"
                    >
                        {{ visible.percent_change }}%
                    </a>
                </span>
                <span
                    v-bind:class="{
                        price: true,
                        dollars: true,
                        negative: visible.negative,
                    }"
                    v-if="visible.price"
                >
                    ${{ visible.price }}
                </span>
                <span class="basis" v-if="visible.position">
                    <span class="units">B</span>
                    <span>${{ visible.position.basis_price }}</span>
                </span>
                <span class="total-value" v-if="visible.position">
                    <span class="units">T</span>
                    <span>${{ visible.position.total_value }}</span>
                </span>
                <span class="portfolio-value" v-if="visible.position && portfolioValue > 0">
                    <span>{{ ((visible.position.total_value / portfolioValue) * 100).toFixed(1) }}</span>
                    <span class="units">%</span>
                </span>
                <span class="portfolio-value" v-if="visible.position && portfolioValue != visibleValue">
                    <span>{{ ((visible.position.total_value / visibleValue) * 100).toFixed(1) }}</span>
                    <span class="units">%V</span>
                </span>
                <span class="factors" v-if="notesFactor">{{ notesFactor.toFixed(2) }}</span>
                <span class="btn-group btn-group-sm" role="group">
                    <button
                        type="button"
                        class="btn btn-outline-primary"
                        v-if="this.visible.has_candles"
                        v-on:click="changeDuration('2D')"
                        v-bind:class="{
                            active: effectiveDuration == '2D',
                        }"
                    >
                        2D
                    </button>
                    <button
                        type="button"
                        class="btn btn-outline-primary"
                        v-on:click="changeDuration('3M')"
                        v-bind:class="{
                            active: effectiveDuration == '3M',
                        }"
                    >
                        3M
                    </button>
                    <button
                        type="button"
                        class="btn btn-outline-primary"
                        v-on:click="changeDuration('12M')"
                        v-bind:class="{
                            active: effectiveDuration == '12M',
                        }"
                    >
                        1Y
                    </button>
                </span>
                <span class="btn-group btn-group-sm" role="group">
                    <button
                        type="button"
                        class="btn btn-outline-primary"
                        v-on:click="toggleOption('MACD')"
                        v-bind:class="{
                            active: isOptionEnabled('MACD'),
                        }"
                    >
                        MACD
                    </button>
                    <button
                        type="button"
                        class="btn btn-outline-primary"
                        v-on:click="toggleOption('ADI')"
                        v-bind:class="{
                            active: isOptionEnabled('ADI'),
                        }"
                    >
                        ADI
                    </button>
                    <button
                        type="button"
                        class="btn btn-outline-primary"
                        v-on:click="toggleOption('50MA')"
                        v-bind:class="{
                            active: isOptionEnabled('50MA'),
                        }"
                    >
                        50MA
                    </button>
                    <button
                        type="button"
                        class="btn btn-outline-primary"
                        v-on:click="toggleOption('100MA')"
                        v-bind:class="{
                            active: isOptionEnabled('100MA'),
                        }"
                    >
                        100MA
                    </button>
                    <button
                        type="button"
                        class="btn btn-outline-primary"
                        v-on:click="toggleOption('200MA')"
                        v-bind:class="{
                            active: isOptionEnabled('200MA'),
                        }"
                    >
                        200MA
                    </button>
                </span>
                <span class="options-chain">
                    <a :href="optionsChainUrl" target="_blank">chain</a>
                </span>
                <span class="tag" v-for="tag in visible.tags" v-bind:key="tag">
                    <router-link :to="'/tags/' + tag">#{{ tag }}</router-link>
                </span>
                <span class="links" v-if="visible.info">
                    {{ visible.info.name.substring(0, 50) }}
                </span>
                <span class="version" v-if="false">{{ version }}</span>
            </div>
            <Chart
                :symbol="symbol"
                :duration="effectiveDuration"
                :chartW="scaleChartW"
                :chartH="chartH"
                :theme="theme"
                :version="version"
                :options="options"
                @click="onChartClick"
                @double-click="onChartDoubleClick"
            />
        </div>
        <div v-if="expanded" class="expansion">
            <form class="form-inline">
                <textarea class="form-control" id="notes" rows="8" v-model="form.notes"></textarea>
                <button class="btn btn-primary" v-on:click.prevent="onSave()">Save</button>
            </form>
        </div>
    </div>
</template>

<script>
import _ from "lodash";
import { makeApiUrl } from "./api";
import Chart from "./Chart.vue";

export function getNotesFactor(s) {
    if (s.noted_prices.length == 0) {
        return null;
    }
    const np = Number(s.noted_prices[0]);
    const diff = -(np - s.price);
    return (diff / s.price) * 100;
}

export default {
    name: "SymbolExplorer",
    components: {
        Chart,
    },
    props: {
        symbol: {
            type: String,
            required: true,
        },
        info: {
            type: Object,
            required: true,
        },
        chartW: {
            type: Number,
            required: true,
        },
        chartH: {
            type: Number,
            required: true,
        },
        scale: {
            type: Number,
            default: 1,
        },
        expanded: {
            type: Boolean,
            default: false,
        },
        portfolioValue: {
            type: Number,
            required: true,
        },
        visibleValue: {
            type: Number,
            required: true,
        },
        theme: {
            type: String,
            required: true,
        },
        columns: {
            type: Number,
            required: true,
        },
    },
    data() {
        const notes = this.info.notes.length > 0 ? this.info.notes[0].body : "";
        return {
            duration: "3M",
            options: [],
            form: {
                notes: notes,
            },
            visible: this.info,
        };
    },
    computed: {
        classes() {
            return [
                "symbol-container",
                "stock",
                this.symbol,
                this.info.column >= 2 ? "right" : "left",
                this.visible.position ? "holding" : "watching",
                this.expanded ? "expanded" : "collapsed",
                this.columns == 1 ? "monocolumn" : "multicolumn",
            ].join(" ");
        },
        tagClasses() {
            return this.visible.tags.map((t) => t.replace(":", "-")).join(" ");
        },
        stockChartsUrl() {
            return `https://stockcharts.com/acp/?s=${this.symbol}`;
        },
        fidelityUrl() {
            return `https://digital.fidelity.com/prgw/digital/research/quote/dashboard?symbol=${this.symbol}`;
        },
        seekingAlphaUrl() {
            return `https://seekingalpha.com/symbol/${this.symbol}`;
        },
        optionsChainUrl() {
            return `https://researchtools.fidelity.com/ftgw/mloptions/goto/optionChain?cusip=&symbol=${this.symbol}&Search=Search&symbols=${this.symbol}&showsymbols=N&sortBy=EXDATE_EXTYPE_OPTYPE_ADJ`;
        },
        effectiveDuration() {
            return this.duration;
        },
        scaleChartW() {
            if (this.scale <= 2) {
                return this.chartW * this.scale;
            }
            return this.chartW * 2;
        },
        version() {
            return this.visible.version;
        },
        notesFactor() {
            return getNotesFactor(this.visible);
        },
    },
    watch: {
        expanded(after, before) {
            console.log("expanded-change", before, after);
            if (after) {
                if (this.duration == "3M") {
                    this.duration = "12M";
                }
            } else {
                if (this.duration == "12M") {
                    this.duration = "3M";
                }
            }
        },
    },
    methods: {
        onChartClick() {
            console.log("chart-click", this.symbol);

            this.$emit("click");
        },
        onChartDoubleClick() {
            console.log("chart-double-click", this.symbol);

            this.$emit("double-click");
        },
        async onSave() {
            this.$emit("saved");

            const post = {
                body: this.form.notes,
                notedPrice: this.info.price,
            };
            const response = await fetch(makeApiUrl(`/symbols/${this.symbol}/notes`), {
                method: "POST",
                body: JSON.stringify(post),
            });
            const stock = await response.json();
            this.visible = stock;

            console.log("stock", stock);
        },
        async changeDuration(duration) {
            this.duration = duration;
            console.log("change-duration:", duration);
        },
        isOptionEnabled(name) {
            return _.indexOf(this.options, name) >= 0;
        },
        toggleOption(name) {
            if (this.isOptionEnabled(name)) {
                this.options = _.without(this.options, name);
            } else {
                // TODO When we add more options this will screw things up.
                this.options = [name];
            }
        },
    },
};
</script>
<style scoped>
.stock .header {
    min-height: 110px;
    display: flex;
    flex-direction: row;
    justify-content: flex-start;
    align-content: flex-start;
    align-items: baseline;
    padding-top: 0em;
    padding-bottom: 0em;
    padding-left: 0.5em;
    padding-right: 4.5em;
    font-weight: bold;
    flex-wrap: wrap;
    padding: 10px;
}

.stock .header > div,
.stock .header > span {
    padding-right: 0.35em;
}

.stock .header .links {
    font-size: 9pt;
}

.stock .header .version {
    font-size: 7pt;
    color: #afafaf;
}

.stock .price {
    font-size: 10pt;
    font-weight: bold;
    color: #0a7b44;
}

.stock .price.negative {
    color: #c9392c;
}

.stock .percent {
    font-size: 15pt;
}

.stock .symbol a {
    font-size: 16pt;
    font-weight: bold;
}

.stock .basis {
    font-size: 10pt;
    color: #808080;
}

.stock .basis .units {
    color: #808080;
    opacity: 0.4;
}

.stock .total-value {
    font-size: 10pt;
    color: #808080;
}

.stock .total-value .units {
    color: #808080;
    opacity: 0.4;
}

.stock .units {
    font-size: 10pt;
    opacity: 0.4;
}

.stock .expanded {
    width: 100%;
}

.stock .tag a {
    color: #afafaf;
    color: #8c5e80;
}

.stock .show-candles,
.stock .show-daily {
    color: #808080;
    font-weight: bold;
    font-size: 14pt;
    cursor: pointer;
}

/* Status Bar */

.status {
    height: 10px;
    padding: 4px;
}
.status div {
    height: 8px;
}

#app.dark .status.v-hold div {
    background-color: #202020;
}
#app.dark .status.v-old div {
    background-color: #312700;
}

#app.light .status.v-hold div {
    background-color: #f0f0f0;
}
#app.light .status.v-old div {
    background-color: #fff4ca;
}

#app.paper .status.v-hold div {
    background-color: #ffdfc4;
}
#app.paper .status.v-old div {
    background-color: #fff4ca;
}

.form-inline {
    width: 100%;
    height: 100%;
    padding: 1em;
}

.form-inline textarea {
    width: 100%;
    margin: 0em 0em 0em 0em;
    margin-bottom: 1em;
    background-color: #c0c0c0;
}

.symbol-container {
    display: flex;
    flex-direction: row;
    align-items: flex-end;
}

.symbol-container .btn-group .btn {
    padding: 1px 3px 1px 3px;
}

.symbol-container.expanded {
    width: 100%;
}

.symbol-container .expansion {
    width: 100%;
}

.symbol-container.multicolumn .expansion {
    height: 100%;
}

.symbol-container.expanded.right {
    flex-direction: row-reverse;
}

.symbol-container.expanded.right .form-inline {
    justify-content: flex-end;
}

.symbol-container.monocolumn {
    flex-direction: column;
}

.symbol-container.multicolumn .expansion {
    display: flex;
    flex-direction: column;
    padding-top: 5em;
}

.symbol-container.multicolumn .expansion .form-inline {
    display: flex;
    flex-direction: column;
    justify-content: flex-start;
    align-content: flex-start;
    align-items: flex-start;
}

.symbol-container.multicolumn.right .expansion .form-inline {
    align-content: flex-end;
    align-items: flex-end;
}

.symbol-container.multicolumn .expansion .form-inline textarea {
    width: 50%;
    flex-grow: 1;
}

/* Debugging */

/*
.expansion {
  background-color: #afaf00;
}
.form-inline {
  background-color: #af00af;
}
*/
</style>
