<script setup lang="ts">
import axios from 'axios';
import type { AxiosResponse, AxiosError } from 'axios';
import ResultBox from './components/ResultBox.vue';
import SearchBar from './components/SearchBar.vue';

function search({query, selector}: {query: string, selector: string}) {
  console.log(query, selector);
  axios.get('/api/search', {
    params: {
      query,
      selector
    }
  }).then((response: AxiosResponse) => {
    console.log(response);
  }).catch((error: Error) => {
    console.log(error);
  });
}

</script>

<template>
  <div class="flex pt-12 flex-col h-screen space-y-4 w-full md:w-1/2 px-8 md:px-0 mx-auto">
    <h1 class="gfont text-6xl md:text-8xl text-center mb-6 space-x-1">
      <p class="inline text-blue-500">T</p>
      <p class="inline text-red-500">w</p>
      <p class="inline text-yellow-500">i</p>
      <p class="inline text-blue-500">t</p>
      <p class="inline text-green-500">a</p>
      <p class="inline text-red-500">l</p>
      <p class="inline text-blue-500">y</p>
      <p class="inline text-red-500">s</p>
      <p class="inline text-yellow-500">i</p>
      <p class="inline text-blue-500">s</p>
    </h1>
    <SearchBar @search="search" />
    <ResultBox :result="{}"/>
  </div>
</template>

<style scoped>

.gfont {
  font-family: 'Product Sans';
}

</style>
