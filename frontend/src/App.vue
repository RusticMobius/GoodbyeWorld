<template id="app">
  <div class="main">
      <router-view/>

  </div>
</template>

<script>
  export default {
    components: {  },
    name: 'App'
  }
</script>

<style>
  /*#app {*/
  /*  font-family: 'Avenir', Helvetica, Arial, sans-serif;*/
  /*  -webkit-font-smoothing: antialiased;*/
  /*  -moz-osx-font-smoothing: grayscale;*/
  /*  text-align: center;*/
  /*}*/
  /*.main{*/
  /*  user-select: none;*/
  /*  !*background: url('./assets/zyb2.jpg') no-repeat center;*!*/
  /*  display: block;*/
  /*  min-height: 100%;*/
  /*  !*box-shadow: 0 2px 6px rgba(0,0,0,.2);*!*/
  /*}*/
  /*.fade-enter-active {*/
  /*  transition: all .3s ease;*/
  /*}*/
  /*.fade-enter, .fade-leave-to{*/
  /*  transform: translateX(20px);*/
  /*  opacity: 0;*/
  /*}*/
</style>
