<template>
    <div id="app" :class="theme">
        <Header>
            <div><a href="#" v-on:click="toggleStyle">theme</a></div>
            <div><a href="#" v-on:click="render">render</a></div>
            <div><a href="#" v-on:click="clear">clear cache</a></div>
            <div class="sort-options">
                sort:
                <a :class="{ selected: sort == sortPriority }" href="#" v-on:click="changeSort(sortPriority)">priority</a>
                <a :class="{ selected: sort == sortSymbol }" href="#" v-on:click="changeSort(sortSymbol)">symbol</a>
                <a :class="{ selected: sort == sortFreshness }" href="#" v-on:click="changeSort(sortFreshness)">freshness</a>
                <a :class="{ selected: sort == sortPercentChange }" href="#" v-on:click="changeSort(sortPercentChange)">% change</a>
                <a :class="{ selected: sort == sortAbsolutePercentChange }" href="#" v-on:click="changeSort(sortAbsolutePercentChange)">
                    % ch abs
                </a>
            </div>
            <div>{{ visible.length }} symbols</div>
            <div>${{ portfolioValue.toFixed(2) }}</div>
            <form class="form-inline">
                <input type="text" v-model="form.filter" />
            </form>
        </Header>
        <div class="tags">
            <span
                :class="{
                    tag: true,
                    selected: tag == tag.tag,
                    empty: tag.n == 0,
                }"
                v-for="tag in tags"
                v-bind:key="tag.tag"
            >
                <router-link :to="'/tags/' + tag.tag">#{{ tag.tag }}</router-link>
                {{ tag.n }}
            </span>
        </div>
        <Symbols :portfolioValue="portfolioValue" :symbols="visible" :market="market" :theme="theme" />
    </div>
</template>

<script>
import _ from "lodash";
import Vue from "vue";
import { makeApiUrl, repeatedly } from "./api";
import { getNotesFactor } from "./Symbol.vue";

import Symbols from "./views/Symbols.vue";
import Header from "./Header.vue";

export const SortPriority = "sortPriority";
export const SortSymbol = "sortSymbol";
export const SortPercentChange = "sortPercentChange";
export const SortAbsolutePercentChange = "sortAbsolutePercentChange";
export const SortFreshness = "sortFreshness";

export default {
    name: "App",
    components: {
        Symbols,
        Header,
    },
    data() {
        const theme = window.localStorage["feevee:theme"];
        const sort = window.localStorage["feevee:sort"];
        return {
            symbols: [],
            market: {
                open: false,
            },
            form: {
                filter: "",
            },
            theme: theme || "dark",
            sort: sort || SortPriority,
            sortSymbol: SortSymbol,
            sortPriority: SortPriority,
            sortPercentChange: SortPercentChange,
            sortAbsolutePercentChange: SortAbsolutePercentChange,
            sortFreshness: SortFreshness,
        };
    },
    computed: {
        tags() {
            const visible = _(this.visible)
                .map((s) => s.tags)
                .flatten()
                .groupBy((t) => t)
                .map((values, tag) => {
                    return { tag: tag, n: values.length };
                })
                .groupBy((row) => row.tag)
                .value();

            const tags = _(this.symbols)
                .map((s) => s.tags)
                .flatten()
                .groupBy((t) => t)
                .map((values, tag) => {
                    const v = visible[tag];
                    if (v) {
                        return { tag: tag, n: v[0].n };
                    }
                    return { tag: tag, n: 0 };
                })
                .sortBy([(row) => row.tag])
                .value();

            return tags;
        },
        tag() {
            const params = this.$route.params;
            if (params && params.tags && params.tags.length > 0) {
                return params.tags;
            }
            return null;
        },
        visible() {
            return _(this.symbols)
                .filter((s) => {
                    return !this.tag || _.indexOf(s.tags, this.tag) >= 0;
                })
                .filter((s) => {
                    return this.form.filter.length == 0 || s.symbol.indexOf(this.form.filter.toUpperCase()) >= 0;
                })
                .value();
        },
        visibleSymbols() {
            return this.visible.map((stock) => stock.symbol);
        },
        portfolioValue() {
            return _(this.symbols)
                .filter((s) => s.position)
                .map((s) => Number(s.position.total_value))
                .sum();
        },
    },
    created() {
        const symbolsUrl = () => {
            if (this.visibleTag) {
                return "/status?tag=" + this.visibleTag;
            }
            return "/status";
        };
        repeatedly(10000, () => {
            return fetch(makeApiUrl(symbolsUrl()))
                .then((response) => response.json())
                .then((data) => {
                    if (this.symbols.length > 0) {
                        const getVersions = (symbols) => {
                            return _(symbols)
                                .map((s) => {
                                    return {
                                        key: `${s.symbol}-${s.version}`,
                                        symbol: s.symbol,
                                    };
                                })
                                .groupBy((r) => r.key)
                                .value();
                        };
                        const before = getVersions(this.symbols);
                        const after = getVersions(data.symbols);
                        const changed = _.differenceBy(after, before, (s) => s.key);
                        if (changed.length > 0) {
                            console.log("changed", changed);
                        }
                    }

                    Vue.set(this, "market", data.market);
                    Vue.set(this, "symbols", this.sortSymbols(data.symbols, this.sort));
                });
        });
    },
    methods: {
        onFilterChange() {
            console.log("filter-change", this.form);
        },
        render() {
            return fetch(makeApiUrl("/render"));
        },
        clear() {
            return fetch(makeApiUrl("/clear"));
        },
        toggleStyle() {
            const themes = ["dark", "light", "paper"];
            const i = _.indexOf(themes, this.theme);
            const n = (i + 1) % themes.length;
            this.theme = themes[n];
            window.localStorage["feevee:theme"] = this.theme;
        },
        changeSort(sort) {
            this.sort = sort;
            Vue.set(this, "symbols", this.sortSymbols(this.symbols, this.sort));
            window.localStorage["feevee:sort"] = this.sort;
        },
        sortSymbols(symbols, sort) {
            switch (sort) {
                case SortFreshness: {
                    return _.sortBy(symbols, [
                        (s) => {
                            return s.freshness;
                        },
                        (s) => {
                            return -Number(s.percent_change);
                        },
                    ]);
                }
                case SortAbsolutePercentChange: {
                    return _.sortBy(symbols, [
                        (s) => {
                            return -Math.abs(Number(s.percent_change));
                        },
                    ]);
                }
                case SortPercentChange: {
                    return _.sortBy(symbols, [
                        (s) => {
                            return -Number(s.percent_change);
                        },
                    ]);
                }
                case SortPriority: {
                    return _.sortBy(symbols, [
                        (s) => {
                            if (s.position) {
                                return 0;
                            }
                            return 1;
                        },
                        (s) => {
                            if (s.noted_prices.length > 0) {
                                return 0;
                            }
                            return 1;
                        },
                        (s) => {
                            if (s.noted_prices.length > 0) {
                                return getNotesFactor(s);
                            }
                            return -Number(s.percent_change);
                        },
                    ]);
                }
            }

            return _.sortBy(symbols, [(s) => s.symbol]);
        },
    },
};
</script>

<style>
#app {
    font-family: Avenir, Helvetica, Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-align: center;
    color: #2c3e50;
    min-height: 100vh;
}

#app.dark {
    background-color: #111111;
}

#app.paper {
    background-color: #fff1e5;
}

#app.light {
    background-color: #ffffff;
}

.top {
    display: flex;
    padding: 10px;
    flex-wrap: wrap;
}
/*
.top .sort-options,
.top .toggles {
  display: flex;
  flex-wrap: wrap;
}
*/

.top div {
    padding-right: 1em;
}

.top .sort-options a,
.top .toggles a {
    padding-right: 1em;
    color: #666666;
}

.top .sort-options .selected,
.top .toggles .selected {
    font-weight: bold;
    color: #8c5e80;
}

.tags {
    display: flex;
    flex-direction: row;
    align-content: flex-start;
    padding: 10px;
    flex-wrap: wrap;
    min-height: 80px;
}

.tag a {
    color: #8c5e80;
}

.tags .tag {
    padding-left: 0.2em;
    padding-right: 0.2em;
}

.tags .tag.selected {
    font-weight: bold;
}

.tags .tag.empty a {
    color: #2c3e50;
}
</style>
