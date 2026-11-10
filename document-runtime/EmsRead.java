import java.io.*;
import java.nio.file.*;
import java.nio.charset.StandardCharsets;
import org.apache.tika.config.TikaConfig;
import org.apache.tika.extractor.EmbeddedDocumentExtractor;
import org.apache.tika.metadata.Metadata;
import org.apache.tika.metadata.TikaCoreProperties;
import org.apache.tika.parser.*;
import org.apache.tika.sax.BodyContentHandler;
import org.xml.sax.ContentHandler;

public class EmsRead {
    public static void main(String[] args) throws Exception {
        Path file=Path.of(args[0]);
        AutoDetectParser parser=new AutoDetectParser(new TikaConfig(Path.of(args[1])));
        ParseContext context=new ParseContext();
        context.set(Parser.class,parser);
        context.set(EmbeddedDocumentExtractor.class,new EmbeddedDocumentExtractor(){
            public boolean shouldParseEmbedded(Metadata metadata){return false;}
            public void parseEmbedded(InputStream stream,ContentHandler handler,Metadata metadata,boolean outputHtml){}
        });
        Metadata metadata=new Metadata();metadata.set(TikaCoreProperties.RESOURCE_NAME_KEY,file.getFileName().toString());
        BodyContentHandler handler=new BodyContentHandler(8*1024*1024);
        try(InputStream input=Files.newInputStream(file)){parser.parse(input,handler,metadata,context);}
        System.out.write(handler.toString().getBytes(StandardCharsets.UTF_8));
    }
}
