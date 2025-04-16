const path = require("path");
const { CleanWebpackPlugin } = require("clean-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");
const BundleAnalyzerPlugin = require("webpack-bundle-analyzer").BundleAnalyzerPlugin;
const Handlebars = require("handlebars");
const fs = require("fs");
const ImageMinimizerPlugin = require("image-minimizer-webpack-plugin");
const HtmlMinimizerPlugin = require("html-minimizer-webpack-plugin");


const FRAGMENTS_PATH = "src/fragments";

// Load the fragments from the fragments directory and caches it
const loadFragmentsMap = (() => {
    let cachedFragments = null;
    return async () => {
        if (cachedFragments === null) {
            cachedFragments = {};
            const walkDir = async (dir, basePath = '') => {
                const files = fs.readdirSync(dir);
                await Promise.all(files.map(async file => {
                    const filePath = path.join(dir, file);
                    const relativePath = path.join(basePath, file);
                    if (fs.statSync(filePath).isDirectory()) {
                        await walkDir(filePath, relativePath);
                    } else {
                        // Remove the .html extension before creating the dotted path
                        const nameWithoutExt = relativePath.replace(/\.html$/, '');
                        const dottedPath = 'fragment-' + nameWithoutExt.replace(/\\/g, '-').replace(/\//g, '-').replace(/\./g, '-');
                        const content = fs.readFileSync(filePath, "utf8");
                        // Minify the HTML content using swcMinifyFragment
                        const minifiedRes = await HtmlMinimizerPlugin.swcMinifyFragment({"tmp.html": content})
                        if (minifiedRes.errors) {
                            console.error(minifiedRes.errors)
                        }
                        const minifiedContent = minifiedRes.code;
                        cachedFragments[dottedPath] = minifiedContent;
                    }
                }));
            };
            await walkDir(FRAGMENTS_PATH);
        }
        return cachedFragments;
    };
})();

const transformHandlebars = async (data, path) => {
    const fragments = await loadFragmentsMap();
    console.log(`Available fragments: ${Object.keys(fragments).join(', ')}`);
    // Load the template file
    const template = Handlebars.compile(data.toString('utf8'));
    const html = template(fragments);
    return html;
};

module.exports = {
    entry: {
        distill: "./src/distill.js",
        main: "./src/index.js",
    },
    output: {
        filename: "[name].bundle.js", // The output file
        path: path.resolve(__dirname, "dist"), // Output directory
    },
    module: {
        rules: [
            { test: /\.css$/, use: ["style-loader", "css-loader"] },
            {
                test: /\.(js|mjs)$/,
                exclude: /node_modules/,
                use: {
                    loader: "babel-loader",
                    options: {
                        presets: ["@babel/preset-env"],
                    },
                },
            },
            {}
        ],
    },
    plugins: [
        new CleanWebpackPlugin(),
        new CopyPlugin({
            patterns: [
                {
                    from: "assets",
                    to: "assets",
                },
                { from: "src/fragments/*", to: "fragments/[name].html" },
                { from: "src/style.css", to: "style.css" },
                { from: "src/bibliography.bib", to: "bibliography.bib" },
                { 
                    from: "src/index.html", 
                    to: "index.html",
                    transform: transformHandlebars, 
                },
            ],
        }),
    ],
    devtool: process.env.NODE_ENV === 'production' ? 'source-map' : 'eval-source-map',
    devServer: {
        static: "./dist", // Serve files from the 'dist' directory
        open: process.env.NODE_ENV !== 'production', // Automatically open the browser unless in production
        hot: process.env.NODE_ENV !== 'production', // Enable hot module replacement unless in production
    },
    mode: process.env.NODE_ENV === 'production' ? 'production' : 'development',
    optimization: {
        minimizer: [
            // Hynek: Ideally we would convert all images to webp and just use webp, but I don't have time
            // to write script which would also modify html to reflect the new extensions.
            new ImageMinimizerPlugin({
                minimizer: [{
                    implementation: ImageMinimizerPlugin.sharpMinify,
                    options: {
                        encodeOptions: {
                            // For JPG
                            jpeg: {
                                quality: 80
                            },
                            // For PNG
                            png: {
                                quality: 80
                            },
                            // For WebP
                            webp: {
                                quality: 80
                            }
                        }
                    }
                },
                {
                    implementation: ImageMinimizerPlugin.svgoMinify,
                    options: {
                        encodeOptions: {
                            multipass: true,
                            plugins: [
                                'preset-default',
                            ]
                        }
                    }
                }
            ]
            }),
            //Hynek: Ideally we don't run this twice but we
            new HtmlMinimizerPlugin({
                test: /fragments\/.*\.html$/i,
                minify: HtmlMinimizerPlugin.swcMinifyFragment,
            })
        ]
    },
};

console.log(process.env.NODE_ENV)