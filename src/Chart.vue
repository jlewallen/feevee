<template>
  <div class="symbol-chart" ref="chartTop">
    <img
      v-lazy="visible"
      :width="chartW"
      :height="chartH"
      v-on:click="onClick()"
      v-on:dblclick="onDoubleClick()"
      v-if="visible"
      v-show="loaded"
      @load="onLoaded"
    />
    <div
      v-show="!visible || !loaded"
      :style="{ width: chartW + 'px', height: chartH + 'px' }"
    ></div>
  </div>
</template>

<script>
import { makeApiUrl } from "./api";

const makeUrl = (chart) => {
  if (chart.chartW == 0 || chart.chartH == 0)
    throw new Error("invalid chart dimensions");
  if (chart.candles) {
    return makeApiUrl(
      `/symbols/${chart.symbol}/candles/${chart.days}/${chart.chartW}/${chart.chartH}/${chart.theme}?c=${chart.version}`
    );
  }
  return makeApiUrl(
    `/symbols/${chart.symbol}/ohlc/${chart.months}/${chart.chartW}/${chart.chartH}/${chart.theme}?c=${chart.version}`
  );
};

export default {
  name: "Chart",
  data() {
    return {
      loaded: false,
      visible: makeUrl({
        symbol: this.symbol,
        candles: this.candles,
        chartW: this.chartW,
        chartH: this.chartH,
        months: this.months,
        days: this.days,
        version: this.version,
        theme: this.theme,
      }),
    };
  },
  props: {
    symbol: {
      type: String,
      required: true,
    },
    months: {
      type: Number,
      default: 3,
    },
    days: {
      type: Number,
      default: 2,
    },
    chartW: {
      type: Number,
      default: 300,
    },
    chartH: {
      type: Number,
      default: 300,
    },
    version: {
      type: String,
      default: "",
    },
    candles: {
      type: Boolean,
      default: false,
    },
    theme: {
      type: String,
      default: "dark",
    },
  },
  computed: {
    url() {
      return makeUrl({
        symbol: this.symbol,
        candles: this.candles,
        chartW: this.chartW,
        chartH: this.chartH,
        months: this.months,
        days: this.days,
        version: this.version,
        theme: this.theme,
      });
    },
  },
  watch: {
    url() {
      const loading = this.url;
      const image = new Image();
      this.visible = null;
      console.log("loading", loading);
      image.onload = () => {
        this.visible = loading;
      };
      image.src = loading;
    },
  },
  methods: {
    onClick() {
      this.$emit("click", this.symbol);
    },
    onDoubleClick() {
      this.$emit("double-click", this.symbol);
    },
    onLoaded() {
      this.loaded = true;
    },
  },
};
</script>
<style scoped></style>
