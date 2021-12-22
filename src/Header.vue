<template>
    <div class="top">
        <div><a href="#" v-on:click="onHome">home</a></div>
        <div><router-link to="/profile">profile</router-link></div>
        <div><router-link to="/options">options</router-link></div>
        <slot name="application-links"></slot>
        <div><a class="outside" href="https://money.cnn.com/data/fear-and-greed/" target="blank">f&g index</a></div>
        <div><a class="outside" href="https://stockcharts.com/def/servlet/Favorites.CServlet" target="_blank">sc/dash</a></div>
        <div><a class="outside" href="https://stockcharts.com/def/servlet/SC.scan" target="_blank">sc/scans</a></div>
        <slot></slot>
    </div>
</template>

<script>
export default {
    name: "Header",
    methods: {
        onHome() {
            if (this.$route.path != "/") {
                this.$router.replace("/");
            }
            this.$emit("nav-home");
        },
    },
};
</script>

<style>
.top {
    display: flex;
    padding: 10px;
    flex-wrap: wrap;
}

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

.top a.outside {
    color: #000000;
}

#app.dark .top a.outside {
    color: #ffffff;
}
</style>
