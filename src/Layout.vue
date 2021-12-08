<template>
  <div id="app" :class="theme">
    <router-view></router-view>
  </div>
</template>

<script>
// import _ from "lodash";

export default {
  name: "Layout",
  data() {
    const theme = window.localStorage["feevee:theme"];
    return {
      theme: theme || "dark",
    };
  },
  components: {},
  computed: {},
  methods: {},
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

.top div {
  padding-right: 1em;
}
</style>
