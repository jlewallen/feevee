<template>
    <div id="app">
        <div class="top">
            <div><router-link to="/">home</router-link></div>
        </div>
        <div class="editors">
            <div class="symbols-editor">
                <h4>Symbols</h4>
                <form class="form">
                    <div class="form-group">
                        <textarea class="form-control" rows="8" v-model="form.symbols"></textarea>
                    </div>
                    <div class="buttons">
                        <button class="btn btn-primary" v-on:click.prevent="onAddRemoveSymbols(true)">Save</button>
                        <button class="btn btn-danger" v-on:click.prevent="onAddRemoveSymbols(false)">Remove</button>
                    </div>
                </form>
            </div>
            <div class="lots-editor">
                <h4>Lots</h4>
                <form class="form">
                    <div class="form-group">
                        <textarea class="form-control" rows="8" v-model="form.lots"></textarea>
                    </div>
                    <div class="buttons">
                        <button class="btn btn-primary" v-on:click.prevent="onSaveLots()">Save</button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</template>

<script>
import { makeApiUrl } from "./api";

export default {
    name: "SymbolsEditor",
    data() {
        return {
            form: {
                symbols: "",
                lots: "",
            },
        };
    },
    components: {},
    computed: {},
    methods: {
        async onAddRemoveSymbols(adding) {
            const symbols = this.form.symbols
                .split(/[\s,]/)
                .filter((s) => s.length > 0)
                .map((s) => s.toUpperCase());

            const options = {
                method: "POST",
                body: JSON.stringify({ adding, symbols }),
            };

            const response = await fetch(makeApiUrl("/symbols"), options);
            const data = await response.json();
            console.log("symbols", data);

            this.$router.push("/");
        },
        async onSaveLots() {
            const options = {
                method: "POST",
                body: JSON.stringify({ lots: this.form.lots }),
            };

            const response = await fetch(makeApiUrl("/lots"), options);
            const data = await response.json();
            console.log("save-lots", data);

            this.$router.push("/");
        },
    },
};
</script>

<style>
.editors {
    display: flex;
    flex-direction: column;
    padding: 1em;
}

.editors h3,
.editors h4 {
    text-align: left;
}

.editors .buttons {
    display: flex;
    justify-content: space-between;
    margin-bottom: 1em;
}
</style>
