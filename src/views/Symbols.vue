<template>
    <div class="symbols" ref="symbols">
        <SymbolExplorer
            v-for="(symbol, index) in ordered"
            v-bind:key="`${symbol.symbol}-${symbol.version}`"
            :symbol="symbol.symbol"
            :info="symbol"
            :chartW="chartW"
            :chartH="300"
            :scale="expandedScale(symbol.symbol)"
            :expanded="expansions[symbol.symbol]"
            :portfolioValue="portfolioValue"
            :visibleValue="visibleValue"
            :theme="theme"
            :columns="columns"
            @click="() => onClick(symbol, index)"
            @saved="() => onClick(symbol, index)"
            @expansion="(expanded) => onExpansion(symbol, index, expanded)"
        />
    </div>
</template>

<script>
import _ from "lodash";
import Vue from "vue";
import SymbolExplorer from "../Symbol.vue";

export default {
    name: "Symbols",
    components: {
        SymbolExplorer,
    },
    props: {
        symbols: {
            type: Array,
            required: true,
        },
        market: {
            type: Object,
            required: true,
        },
        theme: {
            type: String,
            required: true,
        },
        portfolioValue: {
            type: Number,
            required: true,
        },
    },
    data() {
        return {
            width: 0,
            expansions: {},
        };
    },
    mounted() {
        this.width = this.$refs.symbols.clientWidth;
    },
    computed: {
        columns() {
            if (this.width > 2400) {
                return 8;
            }
            if (this.width > 1200) {
                return 4;
            }
            if (this.width > 500) {
                return 2;
            }
            return 1;
        },
        chartW() {
            if (this.columns == 1) {
                return this.width;
            }
            return Math.trunc(this.width / this.columns) - 5;
        },
        visibleValue() {
            return _(this.symbols)
                .map((s) => s.position)
                .compact()
                .map((s) => Number(s.total_value))
                .sum();
        },
        ordered() {
            const ordered = _.clone(this.symbols);

            console.log("ordered:calculate");

            const columns = this.columns;
            if (columns > 1) {
                const getWidth = (symbol) => {
                    return this.expansions[symbol] ? this.columns : 1;
                };

                let width = 0;
                for (let i = 0; i < ordered.length; ++i) {
                    const column = width % columns;
                    const symbol = ordered[i];
                    symbol.column = column; // Unhappy about this.

                    const symbolWidth = getWidth(ordered[i].symbol);
                    if (symbolWidth > 1) {
                        console.log(
                            "symbol-width",
                            "columns",
                            columns,
                            symbol.symbol,
                            "column",
                            column,
                            "symbol-width",
                            symbolWidth,
                            "width",
                            width,
                            "mod",
                            (width + symbolWidth) % columns
                        );
                        if (column >= 1) {
                            const swap = ordered[i];
                            ordered[i] = ordered[i - column];
                            ordered[i - column] = swap;
                        }
                    }

                    width += symbolWidth;
                }
            }
            return ordered;
        },
    },
    methods: {
        expandedScale(symbol) {
            if (this.columns <= 1) {
                return 1;
            }
            return this.expansions[symbol] ? 4 : 1;
        },
        onClick(symbol) {
            Vue.set(this.expansions, symbol.symbol, !this.expansions[symbol.symbol]);
        },
    },
};
</script>
<style scoped>
.symbols {
    display: flex;
    flex-wrap: wrap;
}

/*
.symbols .holding,
.symbols .watching {
  border-left: 1px solid transparent;
}

.symbols > .holding + .watching {
  border-left: 1px solid #404040;
}
*/
</style>
