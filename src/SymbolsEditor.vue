<template>
  <div id="app">
    <div class="top">
      <div><router-link to="/">home</router-link></div>
    </div>
    <div class="symbols-editor">
      <form class="form">
        <div class="form-group">
          <textarea
            class="form-control"
            rows="8"
            v-model="form.symbols"
          ></textarea>
        </div>
        <div class="buttons">
          <button class="btn btn-primary" v-on:click.prevent="onSave(true)">
            Save
          </button>
          <button class="btn btn-danger" v-on:click.prevent="onSave(false)">
            Remove
          </button>
        </div>
      </form>
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
      },
    };
  },
  components: {},
  computed: {},
  methods: {
    async onSave(adding) {
      const symbols = this.form.symbols
        .split(/[\s,]/)
        .filter((s) => s.length > 0)
        .map((s) => s.toUpperCase());
      console.log("save", this.form);
      console.log("save", symbols);

      const options = {
        method: "POST",
        body: JSON.stringify({ adding, symbols }),
      };

      const response = await fetch(makeApiUrl("/symbols"), options);
      const data = await response.json();
      console.log("save", data);

      this.$router.push("/");
    },
  },
};
</script>

<style>
.symbols-editor {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
}

.buttons {
  display: flex;
  justify-content: space-between;
}
</style>
