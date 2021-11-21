<template>
  <div :class="classes">
    <div class="main">
      <div :class="'status ' + tagClasses"><div></div></div>
      <div class="header" v-bind:style="{ width: scaleChartW + 'px' }">
        <div class="symbol">
          <a :href="fidelityUrl" target="_blank">{{ symbol }}</a>
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
            >{{ visible.percent_change }}%</a
          >
        </span>
        <span
          v-bind:class="{
            price: true,
            dollars: true,
            negative: visible.negative,
          }"
          >${{ visible.price }}</span
        >
        <span class="basis" v-if="visible.position"
          ><span class="units">B</span>${{ visible.position.basis_price }}</span
        >
        <span class="total-value" v-if="visible.position"
          ><span class="units">T</span>${{ visible.position.total_value }}

          <span class="portfolio-value">
            {{
              ((visible.position.total_value / portfolioValue) * 100).toFixed(
                1
              )
            }}<span class="units">%</span></span
          >
          <span class="portfolio-value" v-if="portfolioValue != visibleValue">
            {{ ((visible.position.total_value / visibleValue) * 100).toFixed(1)
            }}<span class="units">%V</span></span
          >
        </span>
        <span class="factors" v-if="notesFactor">{{
          notesFactor.toFixed(2)
        }}</span>
        <span
          class="show-candles"
          v-if="this.visible.has_candles && !shouldShowCandles()"
          v-on:click="showingCandles = true"
          >C</span
        >
        <span
          class="show-daily"
          v-if="this.visible.has_candles && shouldShowCandles()"
          v-on:click="showingCandles = false"
          >D</span
        >
        <span class="tag" v-for="tag in visible.tags" v-bind:key="tag">
          <router-link :to="'/' + tag">#{{ tag }} </router-link>
        </span>
        <span class="links" v-if="visible.info">
          {{ visible.info.name.substring(0, 50) }}
        </span>
        <span class="version" v-if="false"> {{ version }} </span>
      </div>
      <Chart
        :symbol="symbol"
        :months="months"
        :chartW="scaleChartW"
        :chartH="chartH"
        :theme="theme"
        :version="version"
        :candles="shouldShowCandles()"
        @click="onChartClick"
        @double-click="onChartDoubleClick"
      />
    </div>
    <div v-if="expanded" class="expansion">
      <form class="form-inline">
        <textarea
          class="form-control"
          id="notes"
          rows="8"
          v-model="form.notes"
        ></textarea>
        <button class="btn btn-primary" v-on:click.prevent="onSave()">
          Save
        </button>
      </form>
    </div>
  </div>
</template>

<script>
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
    showCandles: {
      type: Boolean,
      required: false,
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
      form: {
        notes: notes,
      },
      visible: this.info,
      showingCandles: null,
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
    seekingAlphaUrl() {
      return `https://seekingalpha.com/symbol/${this.symbol}`;
    },
    fidelityUrl() {
      return `https://digital.fidelity.com/prgw/digital/research/quote/dashboard?symbol=${this.symbol}`;
    },
    months() {
      return this.expanded ? 12 : 4;
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
  methods: {
    onChartClick() {
      console.log("chart-click", this.symbol);

      this.$emit("click");
    },
    onChartDoubleClick() {
      console.log("chart-double-click", this.symbol);

      this.$emit("double-click");
    },
    shouldShowCandles() {
      if (!this.visible.has_candles) {
        return false;
      }
      if (this.showingCandles === null) {
        return this.showCandles;
      }
      return this.showingCandles;
    },
    async onSave() {
      this.$emit("saved");

      const post = {
        body: this.form.notes,
        notedPrice: this.info.price,
      };
      const response = await fetch(
        makeApiUrl(`/symbols/${this.symbol}/notes`),
        {
          method: "POST",
          body: JSON.stringify(post),
        }
      );
      const stock = await response.json();
      this.visible = stock;

      console.log("stock", stock);
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

.stock .header div,
.stock .header span {
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
