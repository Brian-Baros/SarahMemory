/// <reference types="vite/client" />

// vite-imagetools type declarations
declare module '*?format=webp&w=640' {
  const src: string;
  export default src;
}

declare module '*?w=640' {
  const src: string;
  export default src;
}

declare module '*?format=webp' {
  const src: string;
  export default src;
}
