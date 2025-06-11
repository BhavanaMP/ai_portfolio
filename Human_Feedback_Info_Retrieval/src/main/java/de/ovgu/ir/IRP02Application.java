package de.ovgu.ir;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.lucene.queryparser.classic.ParseException;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import de.ovgu.ir.model.DocumentIndexer;
import de.ovgu.ir.model.InformationRetrieval;
import de.ovgu.ir.model.QueryData;
import de.ovgu.ir.model.RerankData;
import lombok.Value;

@SpringBootApplication
public class IRP02Application {
	public static final String _IndexDirectory = "./Index";
		
	public static void main(String[] args) throws IOException, ParseException{
		String filesDirectory = args[0];
		
		 if (Files.notExists(Paths.get(_IndexDirectory))) {
		 Files.createDirectory(Paths.get(_IndexDirectory)); }
		 FileUtils.cleanDirectory(new File(_IndexDirectory));
		 
		 @SuppressWarnings("unused")
		 DocumentIndexer documentIndexer = new DocumentIndexer(_IndexDirectory, filesDirectory);
		 InformationRetrieval ir = new InformationRetrieval(_IndexDirectory);
		 ir.ResetSystem();
			
		 SpringApplication.run(IRP02Application.class, args);
	}

}
