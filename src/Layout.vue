<template>
    <div id="app" :class="theme">
        <router-view :theme="theme" @change-theme="onChangeTheme"></router-view>
    </div>
</template>

<script>
export default {
    name: "Layout",
    data() {
        const theme = window.localStorage["feevee:theme"];
        return {
            theme: theme || "dark",
        };
    },
    methods: {
        onChangeTheme(theme) {
            console.log("theme-changed", theme);
            this.theme = theme;
            window.localStorage["feevee:theme"] = this.theme;
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

.top div {
    padding-right: 1em;
}

.top a {
    color: #0a7b44;
}
</style>
